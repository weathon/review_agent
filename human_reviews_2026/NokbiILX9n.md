# RPNT: Robust Pre-trained Neural Transformer - A Pathway for Generalized Motor Decoding

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Brain decoding aims to interpret and translate neural activity into behaviors. As such, it is imperative that decoding models are able to generalize across variations, such as recordings from different brain sites, distinct sessions, different types of behavior, and a variety of subjects. Current models can only partially address these challenges and warrant the development of pretrained neural transformer models capable to adapt and generalize. In this work, we propose RPNT - Robust Pretrained Neural Transformer, designed to achieve robust generalization through pretraining, which in turn enables effective finetuning given a downstream task. To achieve the proposed architecture of RPNT, we undertook an investigation to determine which building blocks will be suitable for neural spike activity modeling, since components from transformer models developed for other modalities do not transfer directly to neural data.  In particular, RPNT unique components include 1) Multidimensional rotary positional embedding (MRoPE) to aggregate experimental metadata such as site coordinates, session name and behavior types; 2) Context-based attention mechanism via convolution kernels operating on global attention to learn local temporal structures for handling non-stationarity of neural population activity; 3)  Robust self-supervised learning (SSL) objective with uniform causal masking strategies and contrastive representations. We pretrained two separate versions of RPNT on distinct datasets a) Multi-session, multi-task, and multi-subject microelectrode benchmark; b) Multi-site recordings using high-density Neuropixel 1.0 probes. The datasets include recordings from the dorsal premotor cortex (PMd) and from the primary motor cortex (M1) regions of nonhuman primates (NHPs) as they performed reaching tasks. After pretraining, we evaluated the generalization of RPNT in cross-session, cross-type, cross-subject, and cross-site downstream behavior decoding tasks. Our results show that RPNT consistently achieves and surpasses the decoding performance of existing decoding models in all tasks. Our ablation and sweeping analysis demonstrate the necessity and robustness of the proposed novel components.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
To improve motor decoding from neural activity, this paper introduced several new modifications to the standard transformer architecture. Multidimensional Rotary Positional Embedding (MRoPE) allows the model to embed more recording meta data. Context-based attention mechanism uses convolutional kernels on attention maps to capture local temporal structures and handle neural non-stationarity. Uniform random masking across time and neuron dimensions that avoids fixed masking hyperparameters and improves representation learning. RPNT was pretrained on 2 datasets and tested across cross-session, cross-task, cross-subject, and cross-site generalization tasks. It consistently outperformed existing decoding models (e.g., POYO, POSSM).

### Strengths
1. MRoPE and context-based attention are domain-specific modification of the transformer design that directly address neural recording variability and non-stationarity.
2. The experiments cover multiple generalization settings (cross-session, cross-subject, cross-task, cross-site).
3. This paper shows practical utility in low-data regimes, which is relevant for BCIs.

### Weaknesses
1. The authors claimed that this work is a step toward neurofoundation models, but the amount of pretraining data used is relatively limited. Many of the model’s design choices seem motivated by BCI-specific applications, which may limit its applicability to other datasets. A truly promising foundation model should show scalability beyond architectural innovations, yet the paper does not thoroughly explore the scaling behavior of RPNT.

2. The model diagram could be improved for clarity. As it stands, it is difficult to fully understand the model’s components and data flow based only on the figure and its caption.

3. The study focuses exclusively on monkey motor tasks, which restricts its generality. It would be good to test RPNT on other species and task domains, such as the IBL or Allen decision-making datasets. This would test whether the model generalizes beyond motor cortex data, especially given that monkey and mouse neural recordings can have large differences.

### Questions
1. The paper proposes a uniform random masking strategy, which outperforms fixed-ratio masking. I wonder whether this contributes significantly to the decoding performance, since masking can help prevent overfitting (especially when the single- and multi-session datasets are not very large). Could the authors test a similar masking strategy in POYO, which currently uses fixed-ratio coordinated dropout, to see if it improves POYO's decoding performance?

2. The authors state that the local convolutional attention helps address non-stationarity in neural data, but no specific experiments directly test this claim. Could the authors include a cross-day analysis to better motivate this design choice, beyond the existing cross-session evaluations?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a pre-trained neural transformer for motor decoding called RPNT, combing multidimensional rotary embedding (RoPE) to incorporate electrode configurations, context-based attention for non-stationality and SSL objectives based on masked autoencoding and contrastive learning. However, there are key details missing regarding both the model components and experimental setup, which are essential for comprehensive understanding and assessment. Hence I give a rating of non-acceptance.

### Strengths
- Proposed multidimensional RoPE and context-based attention are novel and valuable additions
- Evaluations on two datasets show competitive motor decoding performance
- Ablation studies on the proposed components justifying their importance

### Weaknesses
1. The main result table of the first monkey dataset (Table 1) has all baseline results taken from two previous papers, but based on my understanding the results of RPNT and baselines are not using the same split nor evaluation procedure and nor finetuning strategy, and at least one of the papers was not using the brainset data split that the authors used in this paper. Overall, there may be unfair comparisons.
2. Detailed description on key components of RPNT is lacking, and this makes it difficult to properly assess and compare: 
    - The model takes trial-aligned inputs, but the first monkey dataset has variable number of neurons in each session
    - Context-based attention needs historical data but it is never officially defined
    - Bin size used in the model is not mentioned, making it impossible to judge whether all available outputs are included in the evaluation
    - The model is mentioned as a causal one but the introduction of conv2d layer on attention matrix makes the overall operation non-causal. 
    - Subject similarity in multidimensional RoPE is hand-picked, introducing inductive bias that the evaluation subject is similar to and drawing insights from one of the subjects in the training data

### Questions
Regarding weakness number 1:

1. After a quick check on cited publications, it seems the baseline scores in Table 1 are taken from cited publications POYO & POSSM, where one baseline Wiener Filter is taken from POYO paper and the rest 13 baselines are taken from POSSM paper. Could the authors clarify on this point and if so please provide proper citation to the original paper. Please see the links as below
   - POSSM's Table 1 from: https://arxiv.org/abs/2506.05320v1
   - POYO's Table 2 from: https://proceedings.neurips.cc/paper_files/paper/2023/file/8ca113d122584f12a6727341aaf58887-Paper-Conference.pdf
2. I have experience on the said first monkey datasets, and based on my understanding, the experimental setups in this paper, POYO and POSSM are not the same. As mentioned in the POSSM paper, they used a special causal evaluation to report the results and their splits were not shuffled (as shown in Figure 5 of POSSM paper), and this were supposedly different from the POYO paper, because they both reported POYO-1 results but with different values. In addition, _brainset_ split used by this paper is shuffled (as can be seen in the official implementation), further making it different from the data used in POSSM. Hence I don’t believe it is a fair comparison in Table 1 as RPNT and baselines are likely training and evaluating on different data. Could the authors clarify on what is the exact experimental setup used to run RPNT?
3. In section 4.2 it is claimed that all baselines were using Full-SFT, meaning all available training sessions were used for each finetuning result. I believe that this is incorrect, and only single session data was used for those baselines’ finetuning results.
4. Could authors comment on how the standard deviations were calculated in both tables (marked in the paper as Table 1 and Figure 3)?
5. To my knowledge there are no currently available benchmarking tools for the first monkey dataset, therefore it is very hard to determine whether two results are indeed a fair comparison. I believe it is needed to provide ample evidence that the same setups are being used among multiple papers, including detailed text description and / or code examples, if one want to include the published results.

Regarding weakness number 2:

6. As shown in line 240 page 5, RPNT expects inputs with a fixed number of neurons (as in D), but the first monkey dataset is not trial-aligned, and their individual sessions have different numbers of neurons. Could the author elaborate the preprocessing steps on this dataset to maintain the same of neurons?
7. As mentioned in section 2, the context-based attention takes historical data as input and outputs convolution kernels. This is an important component as indicated in the ablation study, but the historical data is not properly defined thoughout the paper. Please clarify on how to construct the historical data and what are T and T_hist? 
8. Please comment on what are the bin sizes used in the study. RPNT gives one output per bin (mentioned in section 3.5), while in baselines results (POYO & POSSM) all outputs were evaluated. Could authors comment on how to ensure that all outputs are being evaluated in RPNT? Were any aggregation techniques (like averaging) used on the outputs? 
9. It is claimed in the paper that RPNT is doing causal prediction, because of causal masking. However I believe that the introduction of conv2D on attention matrix violates causality constraint, in that the current kv value is affected by both earlier and later kv value. Please correct me if I have misunderstanding. I think it is probably more faithful to the causal operation if switching the order between causal masking and convolution in equation (5). Have the authors tried this variant?
10. For the multidimensional RoPE, as shown in appendix C.4, for the first monkey dataset, 0, 1, 2, 3 are assigned to monkey c, j, m, t, respectively. Because RoPE is a relative positional embedding, this formulation provides an inductive bias that monkey t is more alike to monkey m, compared to monkey c. Could the author comment on why it was chosen in this way? In addition, when more diverse datasets are included in the training, this doesn't seem to be a scalable way, as it needs to hand-pick the relationship between various tasks and various subjects.
11. Was contrastive learning objective only used in the neuropixel dataset? Since it is an important component in loss function, I believe it would be better to describe it in the main text.

Other questions:

12. Since both datasets are monkeys doing similar motor tasks with recording on similar regions, it would be interesting to see how will the model perform when transferring from one to the other?  
13. In neuropixel S17, the train, valid, test splits in use were 0.2, 0.3 and 0.5, could the authors provide insights on why the specific splits? Also it would be interesting to see how the baselines would perform with up to 50% of the data, compared to the proposed method.
14. Please fix the typo in Table 1 (T-RT std of o-POSSM-Mamba & o-POSSM-GRU).
15. The table with neuropixel results is wrongly marked as Figure 3, please correct it.
16. Please define acronym when it is first used, i.e. line 319 page 6.

### Soundness
1

### Presentation
1

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
The authors present a large neural model using a transformer architecture for the purpose of decoding behavior from measured neural activity. In a self-supervised learning phase (SSL), a generalist neural encoder is pretrained to generate neural embeddings. In a subsequent  Supervised Fine Tuning phase (SFT) a readout layer is appended to the neural encoder to predict behaviour. Three main improvements on previous work are detailed. First the authors use a multidimensional version of rotary positional embeddings (MRoPE) where dimensions correspond not only to time as previously done but also to meta-data (recording positions, subject, task). Then the inherent non-stationarity of neural activity is captured using context-based attention mechanisms using convolution kernels. At last the pre-training masking strategy is not fixed but random following a uniform distribution for both time and neuron. 
The model is then put to the bench on extracellular neurophysiology recordings in primates using Utah arrays (dataset A) and multiple Neuropixel insertions (dataset B). The Neuropixel dataset contains multiple sites allowing to test the spatial encoder, while both datasets provide opportunities to test the temporal encoder, multiple sessions, multiple tasks and multiple subjects. The authors present superior R2 decoding results compared baselines in a cross-session, cross-subject and cross-task scenarios for zero-shot, few-shot fine-tuning and full fine-tuning. The zero-shot model presented even matches or exceed fully fine tuned baselines.
Ablation studies show the context based attention mechanism the main driver of improvement, followed by MRoPE and then the pre-training masking strategy.

### Strengths
The concepts are clear and well exposed with the 3 main drivers of performance well motivated: non-stationarity of neural data, multi-dimensional meta-data encoding and the pre-training masking strategy.
The model shows substantial improvement on the baselines, even in a few shot or zero shot regime, on a relatively well-solved decoding task (R2 above 90% for existing baselines).
The ablation studies are useful for a practitioner wanting to explore further neural transformer architectures by ordering which improvement drove the performance the most. 
The limitations of the model are disclosed: the model is pre-trained on a single modality on a narrow range of brain locations.

### Weaknesses
- Authors mention possible future BCI application l. 109, without mentioning computational requirements. It is probable that closed-loop applications will require implementations that have lower computational requirements by orders of magnitude of those large transformers. The claim is particularly dubious on a decoding problem where a simple Wiener filter can achieve a R2 around 0.88. 
- Claims about insights on motor behaviour are not substantiated, here we would like to see further analysis and discussion rather than attention matrix heatmaps in supplementary figures. For example how does the attention matrix analysis outperform good old Pearson correlation analysis (or other functional correlation method) ? What are those neuroscientific insights ? 
- Although the use of m-rope may be novel for large neural models, there is a similar M-Rope  implementation in vision for Qwen2 Peng Wang 2024a. URL https://doi.org/10.48550/arXiv.2409.12191. M stands for multi-modal, but the multi-dimensional block rotation matrix building follows the same pattern as presented by the authors.
- On the public benchmark, no task is "unseen", only the distribution of the task representation in the pre-training data changes. This is a bit misleading, and a proper unseen task experiment may be interesting, especially given the similarity of the protocol (reaching tasks).
- Neither the Neuropixel dataset, nor the model, nor the code are disclosed. 


##### Minor comments:
Figure 1D: typo fintuneing
Figure 1C: add SSL acronym in subcaption

### Questions
What happens if a task is entirely removed from the pre-training set ? This means removing the behaviour component of the M-ROPE.
- What is the Neuropixel POYO / POSM baseline score ? Couldn't we expect above NDT ? 
- Any insight about why the overall R2 decoding scores of Utah dataset cross-subject are so much higher than Neuropixel dataset cross-site ?

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
4

### Summary
The paper presents new transformer model for brain-computer interface decoding of motor tasks. They demonstrate its performance on offline datasets and compare to other transformer models.

### Strengths
- State of the literature was summarized well in this crowded application space. 

- Each model component is motivated, diagramed (Fig. 2), and explained very clearly.  

- Datasets are standard for motor BCI tasks and a fair number of model comparisons were implemented. 

- Limitations were discussed ("tasks represent simplified and stereotyped motor behaviors") appropriately.

### Weaknesses
- Related work was limited to neural decoding and did not explore relevant literature in transformers for time series, non-stationary data in general. 

- The paper seems highly application-specific and driven by offline decoding results. The novelty is weakened by the combining of existing mechanisms (RoPE, masking, context-based attention) for a task-specific model rather than a fundamentally new insight into the performance of transformers for neural data applications. 

- Section 3.2 on context-based attention mechanism wasn't clear on what was novel and what exists already. Attention pooling and convolutional kernels are quite common in transformer applications. 

- Results do not always appear to be significantly different when selecting for best model. Distinguishing between actually different (statistically) and on average slightly better would be better. 

- Offline performance of a decoder does not strictly correlated with online performance. New recordings were done but it is not clear that the model results were collected online. 

- Ablation experiment results were stated without much insight into mechanisms.

### Questions
- How were the comparison models chosen? Why not NDT3 or BrainBERT?

- Which results were significant improvements?

- Is MRoPE is a novel generalization to multiple dimensions? The original paper covers a 2D case and a general form. The text may be misleading to state RoPE was extended to multiple dimensions here without additional qualifiers (line 202).

- Why were these tasks chosen: cross-session, cross-type, cross-subject, and cross-site? Cross-session for generalization across time, type for task, subject for if no training is possible?, site for ...? Aren't different subjects already giving different sites? Or are there different brain regions used for some reason? It seemed to be the same usual regions (PMd, M1).

### Soundness
2

### Presentation
3

### Contribution
2
