# Edit-Based Flow Matching for Temporal Point Processes

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Temporal point processes (TPPs) are a fundamental tool for modeling event sequences in continuous time, but most existing approaches rely on autoregressive parameterizations that are limited by their sequential sampling. Recent non-autoregressive, diffusion-style models mitigate these issues by jointly interpolating between noise and data through event insertions and deletions in a discrete Markov chain. In this work, we generalize this perspective and introduce an Edit Flow process for TPPs that transports noise to data via insert, delete, and substitute edit operations. By learning the instantaneous edit rates within a continuous-time Markov chain framework, we attain a flexible and efficient model that effectively reduces the total number of necessary edit operations during generation. Empirical results demonstrate the generative flexibility of our unconditionally trained model in a wide range of unconditional and conditional generation tasks on benchmark TPPs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces EDITPP, an Edit Flow framework tailored to Temporal Point Processes (TPPs). TPPs model sequences of random events in continuous time, where both the timing and number of events are stochastic—making them a natural fit for the Edit Flow paradigm. Although the training and sampling procedures are largely adapted from the original Edit Flow with minimal changes, applying the framework to TPPs requires several non-trivial modifications. In particular, the authors reformulate the three discrete edit operations—insertion, substitution, and deletion—to operate over continuous event times, effectively bridging discrete edit dynamics with continuous temporal domains. A LLaMA-based architecture serves as the backbone, and the model is evaluated on a range of unconditional and conditional event generation tasks.

### Strengths
The application of Edit Flow to TPP sounds logical and natural.

The adaptation of the framework from operating in a discrete token space to modeling continuous event arrival times is novel, technically sound, and demonstrates a meaningful extension of the original methodology.

Experimental evaluation is extensive.

### Weaknesses
> While natural and flexible, this factorization (referring to autoregression) comes with inherent limitations: sampling scales linearly with sequence length, errors can compound in  multi-step generation, and conditional generation is restricted to forecasting tasks.

The authors claimed the above limitations for autoregressive models but I wonder if the current setup would mitigate these. For example, an autoregression can be understood as (in fact, the Edit Flow authors pointed this out in their paper) all left-to-right insertion operations and in this case, edit based methods don't seem to scale better than AR models wrt sequence length. Clarifications for the other two claims are also needed.

It would be very helpful if the authors could elaborate on why autoregressive models are not favored in this setting since both the data and the task of modeling temporal event sequences are inherently autoregressive to me (if this intuition is wrong, please point out).

### Questions
1. The term “element-wise mixture path” used in Section 3.2 is never explicitly defined in the paper. Should we assume it refers to the same element-wise mixture path introduced in the original Edit Flow paper? If so, it would be helpful to clarify this connection explicitly.

2. Table 1 states "bold is best; underlined second best", but several bolded values differ numerically (like in row MMD, col R/C: 7.5, 6.5, 8.2). Could the authors clarify the criterion for deciding which results are marked as “best”?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduce EDITPP, a new way for modeling temporal point processes (TPPs) using an edit-baesd discrete flow matching approach. The method  not only inserts and deletes events like other models, but also substitutes them. It learns the rates of these edits with a Llama transformer model , and it uses a CTMC framework with a special alignment space to make training possible. 
Because training directly is intractable , the approach uses an auxiliary variable Z, which is an "alignment space". For training, it first samples a noise sequence t_0 and a data sequence t_1 and use the Needleman-Wunsch algorithm to create z_0 and z_1. Then, the model is trained to match the ground-truth edit rates that are defined by this Z alignment path. They test this model on many datasets, both real and synthetic, for tasks like generation and forecasting. It shows that adding the "substitute" operation makes the model more efficient, using fewer edits and running faster than old models like PSDIFF. As a result, EDITPP shows very strong state-of-the-art results on most of the tasks

### Strengths
1. The work demonstrates high-Quality presentation and clarity. It is well-written and effectively uses diagrams to explain its complex methodology.
2. The evaluation of the work is comprehensive. It considers both synthetic and real-world datasets, and both unconditional generation and conditional forecasting tasks. The model also shows much better improved computational efficiency compared to existing diffusion-based TPP models.
3. The paper successfully applies edit flows from discrete text to the more complex continuous and ordered TPP space. The solution—using an auxiliary alignment space (Z) and leveraging the Needleman-Wunsch algorithm for alignment to define a tractable training objective —is a necessary adaptation of edit flows for it to be compatible with temporal point process data.

### Weaknesses
1. Despite being a successful application of edit flow to TPP data, the approach lacks substantial methodological innovation and novelty. There's no essential difference between the approach and the original edit flow and the introduction of the Needleman-Wunsch algorithm for alignment is an innovation specific to TPP data.
2. The evaluation results of EDITPP are weak. For example, in Table 2 for conditional forecasting, EDITPP underperforms the baseline models in the majority of the cases. The author uses bold font to indicate the best performing model and underscore to indicate the second best performing model based on overall ranking in the tables, resulting in multiple bold fonts and underscores in each column. Such presentation of models' overall ranking is redundant, unnecessary and confusing. I strongly advise the author to improve the presentation of Table 2.
3. The proposed approach cannot handle marked TPP data with event categories.
4. Please refer to the questions section.

### Questions
1. Is it possible to extend the edit approach to marked TPP data? For example, can the approach be used to model a TPP for each event category which can be used to compose into one marked TPP? This approach follows existing works like neural Hawkes process [a] that model intensity function for each type of event separately. Alternatively, can we introduce another operation that predict or change event category on top of the insertion, substituting, deleting operations to model marked TPP? Extending the work to marked TPP data could significantly improve the novelty and innovation of the work.
2. Can the approach be adapted to interpolation task in the conditional generation setting? It seems to be a strength of the proposed approach over auto-regressive models like IFTPP.

References:

a. Mei, Hongyuan, and Jason M. Eisner. "The neural hawkes process: A neurally self-modulating multivariate point process." Advances in neural information processing systems 30 (2017).

### Soundness
3

### Presentation
3

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
The paper introduces a novel framework for modeling Temporal Point Processes (TPPs) through continuous-time editing operations. 

The paper presents the EDITPP framework, which integrates random set interpolation methods for TPPs with discrete sequence editing flow methods. This combination allows for a more effective modeling of event sequences.
The framework parameterizes instantaneous rates of insertion, deletion, and substitution operations within a Continuous-Time Markov Chain (CTMC) framework. This approach significantly reduces the number of editing operations required during the generation process.
The experimental results demonstrate that EDITPP achieves state-of-the-art performance on various real-world and synthetic datasets for both unconditional and conditional tasks.
By introducing an auxiliary alignment space, the framework can be flexibly applied to tasks of varying complexity, enhancing its versatility in different scenarios.
Overall, the paper showcases the effectiveness of EDITPP in improving the modeling and generation of event sequences in TPPs.

### Strengths
1. EDITPP effectively combines random set interpolation methods for Temporal Point Processes (TPPs) with discrete sequence editing flow methods, allowing for a more comprehensive modeling approach.
2. By parameterizing the instantaneous rates of insertion, deletion, and substitution within a Continuous-Time Markov Chain (CTMC) framework, EDITPP significantly reduces the number of editing operations required during the generation process, enhancing efficiency.
3. The framework demonstrates state-of-the-art performance on various real-world and synthetic datasets for both unconditional and conditional tasks, showcasing its effectiveness and reliability in practical applications.
4. The introduction of an auxiliary alignment space allows EDITPP to be flexibly applied to tasks of varying complexity, making it versatile for different modeling scenarios and enhancing its applicability across diverse domains.

### Weaknesses
1. While the framework shows strong performance on specific datasets, its generalization capabilities to unseen data or different domains may be limited, necessitating further validation across a broader range of applications.
2. Although EDITPP reduces the number of editing operations, the overall computational efficiency during training and inference may still be a concern, especially when dealing with large-scale datasets or real-time applications.

### Questions
1. Could you provide more detailed guidelines or best practices for hyperparameter tuning? Given that the performance of EDITPP may heavily rely on this aspect, clearer instructions could help practitioners achieve optimal results more efficiently.

2. How does EDITPP perform when applied to datasets that differ significantly from those used in training? Are there any specific limitations observed in terms of generalization?

3. What measures have been taken to ensure the computational efficiency of EDITPP during both training and inference? Are there specific scenarios where the model may struggle with efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a non-autoregressive approach to model temporal point processes (TPP) based on Edit Flows (Havasi, 2025). The proposed method models the rates of edits (insertion, deletion, substitution) via a CTMC. Additionally, it supports modeling sequences of variable lengths by defining an auxiliary alignment space that avoids a separate classifier to determine the number of events during an observation window $[0,T]$.

### Strengths
* The TPP notation is clear and consistent throughout the paper, making the math easier to follow.
* The idea is interesting and novel, especially the design of auxiliary alignment space to support varying lengths of TPPs, which has been one of the key challenges for non-autoregressive approaches.

### Weaknesses
* The idea closely follows the Edit Flows paper for language modeling, from parameterization to writing, e.g., Eq. 9-11 in this paper and Eq. 13-15 in the Edit Flows paper; Eq. 14 in this paper and Eq. 23 in the Edit Flows paper. The main difference between language modeling and TPP is that language modeling uses _discrete_ tokens, while TPPs have _continuous_ timestamps. It is unclear to me how the proposed method handles this. Note that I do not take this as a concern of novelty, but the gap could be better addressed, e.g., there is undefined notation in this paper (borrowed from Edit Flows). Please see more detailed questions below.
* The experimental section is relatively weak. The dataset details are missing, and the paper only uses abbreviations in the main text. The results are also hard to interpret (i.e., a lot of numbers are in bold/underlined). The main text mentions that the ranking is based on the full results in the appendix, but the full results seem to be mixed.
* The relationship between this paper and other non-autoregressive neural TPP papers is less clear, and the paper lacks visualizations on parametric TPP to verify the methodology, e.g., Hawkes processes.

### Questions
I do like the idea, so if the authors could help address the following key aspects in addition to clearer experimental results, I would be happy to support the paper:
1. How did the proposed method adapt the discrete flow in the Edit Flows paper to the continuous variable $t$? More specifically, in line 153, if $n$ goes from 0 to infinity, $t^{(n)}$ could have index 0 and infinity, then how can we have a finite number of events? Could the authors help better explain how the state space is defined, and why it is possible to define the set of _all possible_ padded TPP sequences? What does "padded" refer to?
2. For insertion and substitution, if we use a fixed number of $b_\text{ins}$ and $b_\text{sub}$ per dataset, can the method handle event sequences where the inter-arrival time distribution has a large variance? I do not expect the method to work for all scenarios, but it would be great to better understand the strengths and limitations of the method.
3. For the rate model $u_s^{\theta}$ the paper says they used a Llama backbone (line 309). Is there any specific reason for this modeling choice in the context of TPP?
4. If the interpolated $z_s$ needs to be sorted (line 252-253), does the training take longer?
5. Could the authors elaborate on the "seven real-world and six synthetic benchmark datasets", e.g., which version of the datasets did the paper use, what are the parameters for parametric TPPs, and what are the summary statistics for real-world datasets? It would be better to have the details in a self-contained paper. 
6. Would it also be possible to visualize the ground truth and sampled events for some parametric TPPs and compare different methods, e.g., Hawkes processes and self-correcting processes (if I correctly understand the results in Table 1)?

Other comments:
* $\epsilon$ and $\text{f}_{\text{rm-blanks}}$ are notations in the Edit Flows paper, but not defined in this paper, e.g., it is used in line 231.
* Figure 4, the metric $d_{W_2}$ is missing.
* It would be great to clarify the ranking in results, e.g., taking the average of five random seeds, and also add error bounds to Table 3 for the number of edit operations.

Typo:
* Eq.2: $F(T|\mathcal{H_t})$ -> $F(T|\mathcal{H_{t^{(n)}}})$.

### Soundness
2

### Presentation
3

### Contribution
3
