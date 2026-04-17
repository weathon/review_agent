# Personalized Vision via Visual In-Context Learning

- Decision: Reject
- Scores: 4, 4, 6, 6, 4

## Abstract
Modern vision models, trained on large-scale annotated datasets, excel at predefined tasks but struggle with personalized vision—tasks defined at test time by users with customized objects or novel objectives. Existing personalization approaches rely on costly fine-tuning or synthetic data pipelines, which are inflexible and restricted to fixed task formats. Visual in-context learning (ICL) offers a promising alternative, yet prior methods confine to narrow, in-domain tasks and fail to generalize to open-ended personalization.  We introduce Personalized In-Context Operator (PICO), a simple four-panel framework that repurposes diffusion transformers as visual in-context learners. Given a single annotated exemplar, PICO infers the underlying transformation and applies it to new inputs without retraining. To enable this, we construct VisRel, a compact yet diverse tuning dataset, showing that task diversity, rather than scale, drives robust generalization. We further propose an attention-guided seed scorer that improves reliability via efficient inference scaling. Extensive experiments demonstrate that PICO (i) surpasses fine-tuning and synthetic-data baselines, (ii) flexibly adapts to novel user-defined tasks, and (iii) generalizes across both recognition and generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PICO, a unified framework for visual in-context learning using a four-panel image format and a diffusion-based generative backbone.

### Strengths
- The authors propose a new dataset that consolidates a diverse set of vision tasks into a unified training corpus. This supports the model’s ability to generalise to various task formats under a single image-to-image interface. While the dataset is constructed from existing sources, it is well-structured and contributes to the feasibility of training a universal visual prompting model.

- The model performs competitively across several custom personalisation tasks and evaluates against a broad range of recent baselines.

### Weaknesses
- A good dataset is not merely a collection of existing datasets. It must demonstrate scientific value through structure, diversity, and measurable challenge. While the dataset proposed in PICO is well-organised and supports a wide range of visual tasks, it lacks rigorous analysis of task balance, generalisation limits, or empirical ablations on the selected tasks or images. As such, I do not consider the dataset contribution to be of standalone scientific value.

$~$

- The proposed framework represents an incremental step rather than a fundamentally novel invention. The four-panel in-context format and image-to-image task unification have already been explored in prior works, such as Analogist (Gu et al., 2024, ACM TOG). Similarly, the concept of attention-guided seed selection has appeared in several earlier works and is not introduced here as a fundamentally new technique.

$~$

- The paper's comparisons conflate off-the-shelf models with PICO, which is trained and fine-tuned on a custom dataset. This makes the evaluation setup less fair and diminishes the strength of the comparative claims.

$~$

- The authors argue that their quad-grid format addresses visual task heterogeneity, I do not agree with the strength of this claim (lines 165–168). However, this approach renders all outputs as images which is an effective workaround, but not a principled solution to the problem of heterogeneous output types. It avoids cases that cannot be expressed visually, such as visual question answering or counting (e.g., "how many cats are in the image?"), which the current design cannot accommodate.

$~$

- Although certain benchmarks are held out during training, the underlying task types, segmentation, detection, inpainting, are already present in the training data. Therefore, the evaluations measure generalisation to new instances, not to new task types, weakening the paper's broader generalisation claims.

$~$

- The paper uses a very strong diffusion-based backbone. While this helps showcase what the framework can achieve, without controlling for model size or architectural advantage, it becomes difficult to isolate the impact of the proposed method (and dataset).

$~$

- [Minor] There are several grammar and formatting issues: e.g., Extra “and” in line 209; Two commas in line 211; In general, the writing needs polishing.

$~$

My final remarks here would be that this paper is ambitious and presents a well-engineered system for visual in-context learning. However, its core contributions are built on prior work, with no clear methodological innovation. The performance gains are difficult to attribute given the use of a strong diffusion model and further finetune it, unlike many baselines that rely on off-the-shelf models without even finetuning. Without a disentanglement of these factors, and given that the dataset is primarily a collection of existing sources, I do not find the contribution sufficient for publication.

### Questions
1- How does your framework offer technical novelty beyond prior works?

$~$

2- Given that your model is finetuned on a custom dataset while baselines use off-the-shelf backbones or trained on specific tasks, how do you ensure a fair evaluation?

$~$

3- Can you provide ablations or evidence to disentangle the performance gains from your backbone scale versus the proposed in-context learning mechanism?

$~$

4- What evidence supports the dataset's scientific contribution beyond being a curated aggregation of existing datasets?

$~$

5- If your evaluation benchmarks involve **task types** already present in training (e.g., segmentation), how do you justify claims of generalisation to novel tasks?

$~$

I hope that clear answers to these questions will help clarify the paper's core contributions and allow for a more informed and fair assessment of its scientific merit.

### Soundness
2

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
This paper presents PICO (Personalized In-Context Operator), a framework for personalized vision. The authors reformulate personalization as a visual in-context learning (ICL) problem. In this setting, the model can understand and perform user-defined visual tasks during testing without any extra training or fine-tuning. The method uses a four-panel input format. An example image pair, written as (A → A′), defines the task, and a query image B is given for applying the same transformation, which produces B′. This allows the model to learn the transformation from only one example. To train the system, the authors design a VisRel dataset, which contains a small but diverse set of 27 visual tasks such as image restoration, segmentation, depth estimation, and style transfer. They also use an attention-guided seed scorer that improves the stability and reliability of generative inference by choosing the most effective random seeds.
It can generalize to new user-defined tasks.
It achieves strong results in both recognition and generation tasks with very limited training data.
The VisRel dataset, showing that task diversity is more important than dataset size for strong generalization.
The attention-guided seed scorer, which improves inference consistency without any retraining.

### Strengths
The paper is generally clear and organized, with a logical flow between sections. Figures and diagrams are well designed and help explain the method. However, there are a few small writing and formatting issues in the text, such as spacing and repeated citations, which should be corrected for better readability. 
The experiments are of good quality and cover a range of visual tasks, showing that the model performs consistently across segmentation, generation, and editing. The use of attention-guided seed scorer improves inference stability, even if it is a heuristic approach. The focus is on task variety rather than size, which maybe a good direction for visual in-context learning research.

### Weaknesses
While the paper outlines a diffusion-based visual in-context learning framework, the data flow between components (VAE encoder, diffusion transformer, and LoRA modules) is not clearly explained. The description of the “quad-grid” input format could be supported with more explicit architectural diagrams or pseudocode.
The paper’s main idea of using exemplar pairs (A, A′) and a query image (B) to predict a target (B′) has already been explored in other works, such as "Analogist: Out-of-the-box Visual In-Context Learning with Image Diffusion Model" and "ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation". Similar to Analogist, PICO also use inference-based visual ICL with light text cues and transfer attention relations through diffusion models. Because of these strong similarities, the idea in this paper does not appear to be very novel.
There are minor writing issues, such as no space between vs. and colorful in line 185, and a duplicate citation in lines 113–114 written as Bar et al. (Bar et al., 2022). 


Although the experiments are detailed, the comparison could be clearer if the authors also included results for PRPG. The paper mentions that generating per-instance synthetic data is costly and hard to scale, but adding even limited results for these cases would make the evaluation more complete.
The GPT-4o results are based on only ten random samples due to API limits. Using more samples or a consistent setup would improve fairness and reliability.

### Questions
While PICO builds on similar ideas to Analogist and ImageBrush, which also leverage exemplar pairs and attention transfer for diffusion-based visual in-context learning, could the authors elaborate on the specific innovations introduced by PICO beyond adopting a diffusion transformer in place of the U-Net backbone?

The description of how the VAE encoder, diffusion transformer, and LoRA modules interact is unclear. Could the authors include a more detailed diagram or short pseudocode to show how {A, A′, B} are encoded and passed through the model?

Since PRPG results were omitted due to cost, could the authors share small or partial results to show scalability and fair comparison? 

The GPT-4o evaluation used only ten random samples. How consistent are the results with different samples, and could more data improve reliability?

### Soundness
3

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
This paper considers the problem of efficiently solving personalized vision tasks which often require an unseen, specific visual transformation. The authors propose a simple training algorithm (PICO) with flow-matching to efficiently fine-tune existing generalist models, and they curate the VisRel dataset which contains diverse, structured examples. The paper also proposes an inference method that uses an attention-based scorer to improve sampling quality. The main empirical results show (both qualitatively and quantitatively) that their method generalizes better than baselines, is more sample-efficient, and works well across a variety of tasks and settings.

### Strengths
1. The paper presents several concrete artifacts and ideas that are novel and likely of significance to the community. The most prominent contributions in my view are the VisRel dataset and idea of using an attention-guided seed scorer for inference scaling. 
2. I think the greatest strength of the paper is the empirical rigor. The authors do a thorough job of testing various ablations (task generalization, effect of including text, behavior under various axes of scaling), which strengthens the clarity and quality of the core set of results. The ablations build confidence in the significance of their method. 
3. In addition, I thought the paper did a good job of presentation, including a nice mixture of both quantitative results as well as corresponding qualitative results (which highlight the benefit of their method). The authors clearly and concisely explain the core parts of their method. I also appreciated the analysis in the appendix on the inference scaling method and the one-page view of the full dataset. 
4. Overall, I feel like the quality of the paper is strong, and the paper also includes novel and significant research artifacts and ideas that I believe are of interest to the community.

### Weaknesses
1. I have some reservations about the clarity of the paper with respect to how the paper contextualizes prior work in Visual ICL and how their contribution of framing personalized vision fits within this. See Question 1 for more detail. 
2. There's a lack of clarity explaining some of the baselines in the main experiments as well as other details of the experiments. This results in some confusion when interpreting the experimental results. See Questions 3-5 below for details. Addressing these questions will improve the clarity and quality of the paper.

### Questions
I have various questions about the claims and experimental details. My current rating is a 6, but I'm between a 6 and 8 and I'm willing to increase my score if these are addressed. 

1. I'm a bit confused about the contribution claim of "formulating personalized vision as visual in-context learning". Visual ICL as a setting / method (which is typically task-specific) has been around for a few years. Personalized vision is also defined relative to a distribution of downstream tasks, so I'm missing what exactly the contribution is. To be clear, I think the actual artifact of the VisRel dataset is valuable, but I'd appreciate some clarification on why this is a new formulation. 

2. Is the task taxonomy used in any way during training? Is the main point to highlight the benefit of task diversity? Also, how exactly are the Semantic complexity and Spatial Locality quantified. In 3.1, they seem to be composed of multiple aspects. 

3. I found the main comparison in Table 1 to baselines to be pretty confusing because it was unclear a. which methods were training free b. which methods required more data for training c. which methods were fine-tuned from different models. Adding an additional column classifying this would help a lot for clarity. The table also makes a distinction between large-scale, personalized, generalist methods, but then in Table 2 bottom (labeled large-scale pretrained segmentation models), both large-scale and generalist methods appear. Is the point of Table 2 bottom to show the data efficiency of PICO? At a higher level, this lack of detail on baselines feels more confusing because many of the comparisons in the paper are phrased as "Visual ICL" vs "fine-tuning", but Visual ICL also relies on (a limited amount of) training and some of the generalist methods that you are comparing against are training-based (OmniGen) while others are inference-based (VP). To summarize my questions: 

a. Is Table 2 bottom only to show data efficiency? Do you have thoughts on why your method is much more data efficient? 

b. What are the "win conditions" that you're comparing methods on? Beyond performance, it seems like the set of all methods you tested have different considerations of training-free, how much data is needed, and which models they fine-tune off of. Can you be more clear about these to highlight your wins and show the comparisons are fair? 

c. Do any of the other methods also fine-tune off of FLUX? How much of the wins are coming from this? 

4. Why does performance increase between 12 and 48 shots per task in Figure 6A-a? More generally, why is performance not monotonic in data and task scaling? 
5. Can the TTC method be applied to models that aren't fine-tuned using PICO and would it help in those cases as well?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a concept called the Personalized In-Context Operator (PICO), which allows personalized visual tasks to be carried out through visual in-context learning with a diffusion transformer. The framework is a four-panel framework where single annotated exemplars are presented to the model, the PICO infers the underlying transformations and apply it to the new inputs. In addition to the operator, the paper proposes VisRel, a diverse tuning dataset used to train the model for this task. The work benchmarks their method against an extensive set of experiments, demonstrating that PICO surpasses various baselines and is able to flexibly adapt to various user-defined tasks, spanning both recognition and generation.

### Strengths
- **Flexibility with personalization provided by the ICL framework:** The ICL format of PICO allows this method to be flexible in adapting to new user-defined task without retraining. This is a strong step towards open-ended user-driven visual inference. 
- **Robustness and stability:** The authors propose inference scaling seed selection method to prevent diverging seeds and identifies stable seeds using early cross-attention routing 
- **Diverse Task Generalization:** In addition to standardized personalized detection / segmentation tasks, the authors also test PICO in various flexible tasks that take advantage of the ICL framework - such as composite taks, spatially constrained tasks, and semantic-conditional tasks. The performance shows PICO 
- **Data and Compute Efficiency:** It enables adaptation to new tasks without costly finetuning or model specialization

### Weaknesses
- **Generalization limits:** It is noted that PICO is less reliable on entirely novel task types “outside the trained visual-relation space”. This limits its practical applicability for highly novel or out-of-distribution (OOD) tasks, which are key for real-world personalization
- **Limited demonstration richness:** The current framework only supports one-shot demonstrations, which constrains the expressive power of the provided exemplars. A multi-shot setting could better capture complex or composite transformations.
- **Evaluation coverage:** While the paper presents strong results on diverse benchmarks, most tasks are represented within VisRel or are closely related to it. More systematic evaluation on unseen OOD transformations would better validate the claimed generalization ability.

### Questions
- **Generalization:** How does PICO perform on entirely novel tasks outside of the VisRel taxonomy —  most of the benchmarked datasets were included in VisRel in one form or another (e.g. inpainting, edge detection, segmentation, etc.).  For OOD tasks, is there difference in task performance with respect to other baselines? Could you show some qualitative examples of OOD tasks as well or show benchmarks?
- **Multi-shot ICL:** Could PICO be extended to handle multi-shot or sequential visual demonstrations? Given the reliance on a four-panel grid, how would you handle richer or more structured visual contexts (e.g., videos or multiple exemplars)? Do you expect consistent performance gains with additional exemplars?
- **Task diversity vs. capacity trade-off:** Figure 6 shows that task diversity drives generalization, but increasing the number of tasks hurts in-domain performance. What do you think is a bottleneck to this phenomenon, is it a computational constraint or limitations in the unified visual-relation space itself?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Personalized In-context Operator-  PICO, a framework that applies visual in-context learning to personalized vision tasks. The authors reformulate personalized vision as a visual ICL problem, where a single input-output exemplar defines the task through a four-panel format。 To support this approach, they construct VisRel, a compact training dataset of 27 diverse visual tasks organized by semantic complexity and spatial locality. The method builds upon pretrained diffusion transformers fine-tuned with LoRA, and introduces an attention-guided seed scorer for test-time scaling. Experiments demonstrate that PICO achieves competitive performance on personalized segmentation benchmarks while using significantly less labeled data than baseline methods, and shows the ability to handle user-defined tasks at test time without additional fine-tuning.

### Strengths
- The proposed method demonstrates  improvements over baseline approaches across multiple benchmarks. The attention-guided seed selection strategy is empirically effective.
- The framework successfully handles a diverse range of personalized vision tasks.
- The paper is well-organized and mostly clearly written.

### Weaknesses
Please see questions below for details.
- Critical details regarding the evaluation setup are missing or unclear, which hinders a thorough assessment of the method's effectiveness and generalization capabilities.
- While the authors emphasize that VisRel is compact (315 samples across 27 tasks), the extent to which the method's success depends on having explicit training coverage of test task types remains unclear. This is particularly important given the paper's central claim about generalization to novel, user-defined tasks.

### Questions
- Based on Table 5, most evaluation tasks appear to have corresponding task types in the VisRel training set. Could the authors clarify which evaluation tasks represent truly novel task types not covered during training? Without this clarification, it is difficult to distinguish between interpolation within the trained relation space versus genuine zero-shot generalization to unseen tasks. 
- In Figure 6 and the discussion around line 450, the distinction between "seen" and "unseen" tasks needs clearer definition. Specifically, what constitutes a "seen" versus "unseen" task in this experiment, e.g., novel objects for completely novel tasks?
- In Table 6, and Appendix C.2, why were only 3 tasks removed per category rather than entire task families? Can the authors provide ablations where entire task categories are removed and evaluate on held-out task categories to demonstrate true cross-task generalization?
- What is the computational overhead of the proposed TTS method? Specifically: 1) How does inference time scale with the number of candidate seeds |S|? 2) What is the trade-off between computational budget and performance improvement? 3) Are there diminishing returns beyond a certain number of seeds?

### Soundness
3

### Presentation
3

### Contribution
2
