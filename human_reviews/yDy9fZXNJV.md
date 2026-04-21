# The Graph's Apprentice: Teaching an LLM Low-Level Knowledge for Circuit Quality Estimation

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 6, 6, 3

## Abstract
Logic synthesis is a crucial phase in the circuit design process, responsible for transforming hardware description language (HDL) designs into optimized netlists. However, traditional logic synthesis methods are computationally intensive, restricting their iterative use in refining chip designs. Recent advancements in large language models (LLMs), particularly those fine-tuned on programming languages, present a promising alternative. This work proposes augmenting LLMs with predictor networks trained to estimate circuit quality directly from HDL code. To enhance performance, the model is regularized using embeddings from graph neural networks (GNNs) trained on Look-Up Table (LUT) graphs, thereby incorporating lower-level circuit insights. The proposed method demonstrates superior performance compared to existing graph-based RTL-level estimation techniques on the established benchmark OpenABCD, while providing instant feedback on HDL code quality.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a framework, which utilize a GNN trained on LUT netlist-level graph as a teacher in order to transfer knowledge to LLM encoder based circuit quality estimator. It enables model to get more accurate performance and area of circuits end-to-end with RTL as input.

### Strengths
This paper is the first to implement knowledge transfer of circuit quality estimation from netlist-level to RTL-level. Considering that netlist-level circuit performance and area estimation is more precise and RTL-level circuit quality prediction is faster, this work strike a balance between the precision and inference speed.

### Weaknesses
In this work, the LLM is used as an encoder, leveraging its embeddings to predict circuit quality. However, a potential limitation arises from the token count restrictions of LLMs, which may prevent the framework from handling large Verilog designs. For smaller designs, which can be processed within these token constraints, transforming them into netlists is relatively quick and incurs minimal computational cost. Given this, why not employ a GNN to process the netlist representation directly? Predictions at the netlist level are generally more accurate than those at the RTL level, making this a potentially more effective approach. Moreover, the authors mainly compare there work with models trained on AST. Such an experiment lacks persuasive power.
Things to improve: The ‘Fig. ??’ in section 4.1.

### Questions
1.Given the token count limitations of LLMs, which may prevent the framework from handling large Verilog designs, and the relatively quick and low-cost process of transforming smaller designs into netlists, wouldn’t it be more effective to use a GNN to process the netlist representation directly?
2.Why transform aiger into LUT? What’s the advantage of LUT? And in Advanced integrated circuit (IC) manufacturing process, whether LUT is widely used?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents VeriDistill, the first end-to-end machine learning model that processes raw Verilog code to predict circuit quality metrics. The model uses an innovative knowledge distillation technique, transferring low-level circuit insights via LUT graphs into an LLM-based predictor. Experiments show that VeriDistill surpasses current state-of-the-art methods on large-scale Verilog datasets and demonstrates transferability to out-of-distribution datasets.

### Strengths
This work introduces VeriDistill, an innovative end-to-end machine learning model that predicts circuit quality directly from raw Verilog code, which is an important and challenging task in the domain of electronic design automation (EDA). It applies a novel knowledge distillation method, transferring low-level circuit insights via LUT graphs into an LLM-based predictor. And VeriDistill demonstrates robust performance, outperforming state-of-the-art baselines on large-scale and out-of-distribution datasets.

### Weaknesses
There are several limitations for VeriDistill. Firstly, as shown in Table 1, VeriDistill’s performance lags significantly behind the teacher model. To improve this, I think a possible way is to unfreeze some parameters in the base LLM and fine-tune them with the decoder. Secondly, the paper only uses CodeV as its base model without exploring other Verilog LLMs that might provide different insights. Conducting more experiments with various models could better demonstrate the method's effectiveness and robustness. Additionally, the paper needs careful proofreading; for instance, there is a missing figure referenced in Line 293.

### Questions
1. For Table 1, you state that knowledge distillation has almost no effect on the AST-GNN model. But in Figure 5, you show that the knowledge distillation makes the t-SNE representation of AST-GNN more like that of the teacher model. These statements seem contradictory and confusing. Could you provide further clarification?
2. For Table 2, could you also report the performance of the teacher model on the out-of-distribution dataset? This would help clarify the current gap in performance.
3. It seems VeriDistill can only handle small Verilog designs due to the context window length constraint of the LLM base model. Is there a way to scale VeriDistill for larger designs? Given that logic synthesis for small designs is usually fast and the error remains significant, the current model may not be very attractive.
4. When the logic synthesis recipe changes, the delay and area metrics of circuits typically change as well. How can VeriDistill handle these variations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Introduction of a new ML framework: VeraDistill to predict circuit QoR metrics like area/delay bypassing costly logic synthesis processes.  The framework uses a verilog-trained LLM (codeV-7B) to produce a representation (final layer) to feed into the FFN to make predictions for the QoR metrics during inference.  To train, they take the AIG LUT outputs of the synthesis flow and use GNNs to produce embeddings where the GNN acts as a "teacher" resulting in knowledge distillation. The loss metric used for final training is a simple MSE involving not only the FFN predictions and synthesis QoR data, but also the final layer data from the GCN(teacher) and LLM (Student).  The LLM is frozen in the training process.

### Strengths
As a hardware engineer, I greatly appreciate the contributions in the work as this can provide a lot of benefit to practioners.

I find the knowledge distillation approach between the GNNs and LLMs to be  novel. The work showcases strengths of GNNs and LLMs in a creative fashion. The work also shows that LLMs have remarkably learned about the underlying circuit representation underneath the verilog code.

The results are also impressive, showing improvements over baselines including the out-of-distribution datasets like OpenABCD.

### Weaknesses
Its unclear to me how reliant the framework is on a particular LLM (CodeV) and how versatile a single instance (trained) will be in accommodating more unseen RTL code. The authors do acknowledge the lack of Verilog/RTL code for training, so while not a weakness per se, it does make me question the generalization capabilities.

Not much details are shared about the training resources (outside of the 8 V100 GPUs used).

### Questions
1) Does the knowledge distillation help with a specific subset of Verilog designs (more than others)?  Could be interesting to exploit this.
2) Please elaborate on the training times
3)  There are many ways to code (e.g an adder) in Verilog, does the framework handle this well? 
4) Curious to hear whether you think this approach can scale to handle other RTL (e.g VHDL)

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
4

### Summary
The main contributions of this work, as reported, are:

A. Development of a VeriDistill framework: The first truly end-to-end machine learning model that processes raw Verilog code directly, without preprocessing, to accurately estimate circuit area and delay metrics.

B. Innovative Knowledge Distillation (KD): A novel knowledge distillation method is applied during training, transferring low-level circuit insights (as LUT graphs) back into the model for enhanced predictions. Experiments show that this approach surpasses previous state-of-the-art (SOTA) baselines on a large-scale Verilog dataset, with improved generalization to out-of-distribution (OOD) data.  The use of both LLM representations and knowledge distillation is critical to the model’s performance, as omitting either component decreases performance below baseline levels.

### Strengths
The overall research motivation is well-articulated, and the proposed idea is both interesting and has promising applications.

### Weaknesses
1. The manuscript lacks clear description of the proposed approach. Both figure 1 and figure 2 should be improved in both schematic and description. Methodology section 3.1 must be elaborated. I will advice to break the training strategies and steps at first and finally combine them towards final goal/results. It will enhance readability, understanding of the general audience with more detailed explanation of the intermediate training flows. Figure 1 & 2 and section 3.3 description seems disjointed. Reproducibility from this text description is problematic.

2. in Section 4.1 EXPERIMENTAL SETUP,   "The model’s decoder feedforward network depicted
in Fig. ?? is designed so that the 512-dimensional representations from the...", Fig. reference is missing.

3. The overall quality of writing and presentation could be improved. The writing should maintain a smoother flow, as some paragraphs feel disjointed. For instance, in the experimental setup section 4.1 , certain details—such as the explanation of 512-dimensional state tokens—are important for understanding the methodology and would benefit from clearer integration.

4.  It is recommended to compare the performance of the proposed model with previous models/approaches to better emphasize its novelty, particularly in terms of computational complexity, data preprocessing demands, resource utilization, and the metrics of area and delay. For instance, could you provide an analytical comparison of your approach against the work by Fang et al. (2023/2024b), addressing the computational time for SOG and the complexity of converting linguistic data into bit-level operators using the logic synthesis tool Yosys (Wolf et al., 2013)? This comparison would further clarify the advantages of your approach, especially given that accuracy, precision, and sensitivity are crucial metrics for circuit quality estimation in addition to computational efficiency.

5. In Fig. 4, please clarify the presence of scattered outlier points for both area (lower values) and delay (higher values) in relation to the fitting curve. These sparse data points appear to indicate cases where VeriDistill’s performance aligns with that of other baseline models. Could you explain why VeriDistill performs similarly to the baselines for these specific outliers, and discuss any factors that might contribute to this behavior in terms of model limitations or specific data characteristics?

6. Authors explained "Finally, in Figure 5, we present the t-SNE projection of the last hidden space representations on the test data from the teacher model (Z_teacher) trained for predicting log-area, alongside those from the LLM-based models. As can be seen, the resulting t-SNE representation of the VerDistill model appears very similar to the one of the LUT-GNN teacher model, albeit slightly rotated by roughly 30 degrees to the right (which is arbitrary and immaterial). The AST-GNN with KD model also can be seen as a rotation of the LUT-GNN plot, roughly 90 degrees to the left. In contrast, the plot of the CodeV + Decoder appears much more like an undefined mass." 

It seems hypothetical assumption about the statement "VerDistill model appears very similar to the one of the LUT-GNN teacher model, albeit slightly rotated by roughly 30 degrees to the right (which is arbitrary and immaterial). The AST-GNN with KD model also can be seen as a rotation of the LUT-GNN plot, roughly 90 degrees to the left.".  AST-GNN with KD is quite different than LUT-GNN and VeriDistill, as well it does not seem "t-SNE representation of the VerDistill model appears very similar to the one of the LUT-GNN teacher model"...and if the authors knew already "slightly rotated by roughly 30 degrees to the right".. please adjust this rotation to make it fit overlapped and compared.
""
7. Please be consistent with proposed approach name .......authors mentioned "VerDistill" in one place and "VeriDistill" in other place(s). Please correct any typos. Also, grammatical errors have been noticed.

### Questions
This has already been addressed in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2
