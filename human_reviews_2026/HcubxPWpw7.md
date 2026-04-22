# Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 4

## Abstract
Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies a more complex safety setup than considered before. There are two images and a text given to an MLLM in order to jailbreak it. The idea is that on their own, both images and the text would be without safety concerns, but when reasoning is applied, the combination is unsafe. The paper introduces a dataset that is used for both fine-tuning of pre-trained MLLMs and also evaluating them. Experimental evaluation shows that existing models can be easily jailbroken using the data, but a model fine-tuned using the data is resistant to these kinds of attacks.

### Strengths
•	The paper focuses on a new safety problem setup where the combination of multiple images and a text is unsafe. This is a practically important problem setting as existing models cannot defend themselves well against it and the attack success rate is high.

•	The main contribution of the paper is a dataset that can be used for both fine-tuning of MLLMs and evaluating them for the described form of reasoning-based attack. The pipeline for creating the dataset consists of four stages and seems appropriate for the task. The dataset consists of a few thousand examples.

•	The experimental evaluation shows that existing models fail on the considered problem setting. Fine-tuning on the proposed dataset helps defend against such attacks, and it also helps for general performance. The evaluation uses a solid number of benchmarks for testing both general ability and defence against attacks.

•	A larger number of analyses is performed, and this is helpful for analysing the contribution and for understanding the overall problem setup better.

### Weaknesses
•	The described method MIRage appears to be essentially common fine-tuning on the proposed dataset, so we may not really see it as a specialized new method. Potentially we could say it is VLGuard with different data.

•	The main results seem to only use one model, even if there are some results on other models in the appendix. It would be good to see experiments with more models, e.g. in table 4 in the main text to have a better overview of how the solution works across multiple models.

Minor: small typos such as in L150 "Disccusions", I recommend proof-reading as there are a few more I recall, but not many

### Questions
What exactly is the difference between VLGuard-R and MIRage methods? Could MIRage also be seen as extension of VLGuard?

How computationally expensive was it to create the dataset?

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
4

### Summary
The paper
- Demonstrates two Issues after safety finetuning with VLGuard data
    - Models are prone to overrefusal on benign data, reducing performance on general instruction following data (Table1, left)
    - Models struggle with complex safety situations where compositional dependencies between safe text and safe image can still lead to an unsafe request (Table1, right)
- To address these issues the paper
    - Constructs a new dataset (MIS) where each datapoint consists of text instructions referring to two images and the safety implications of the request requires reasoning about how the two images relate.
    - Generates safe responses for the constructed dataset by prompting InternVL2.5-78B with a general CoT prompt to reason about the inputs and provide a safe response.
    -  Finetunes multiple VL models with the generated training data
- Evaluation includes 
    - multiple existing general VL instruction following and safety tasks. 
    - A new multi-image safety test set from the newly constructed data
- Results show
    - Existing VLMs struggle with providing safe responses to the new multi-image safety data.
    - Models fine-tuned on the new MIS dataset perform almost perfectly on its test set.
    - Models fine-tuned on the new MIS dataset preserve general VL capabilities and improve safety on existing safety datasets.
    - Models fine-tune on the new MIS dataset outperform finetuning on VLGuard labels that were generated in the same way as the MIS labels.

### Strengths
- Construction of the Multi-image safety dataset seems like a valuable contribution especially since many existing models seem to struggle with providing safe responses to the inputs. It is important to be robust against adversarial instructions embedded in multiple images.
- Extensive experiments supporting that improvement from fine-tuning on the new data seems robust across base models and evaluation sets.
- VLGuard-R is a decent baseline and its substantially improved performance over existing methods is interesting in itself.

### Weaknesses
- I am somewhat confused by the results in table 4. How can InternVL2.5 78B perform so much worse than the proposed fine-tuning method if it provided the labels for that fine-tuning in the first place? 
    - This suggests that the evaluation in table 4 does not prompt the models towards providing safe responses. 
    - For a fair comparison, I would want to see the performance of those models when prompted with the same instructions as was done to generate the SFT labels. 
    - This would give a more realistic sense of how impactful the whole dataset creation and fine-tuning procedure was that the paper proposes and to what extend the contribution is contained in the CoT prompt for the label generation. 

I would be likely to support acceptance if this concern is adequately addressed and the claims are adjusted based on the results with the adapted prompt.

### Questions
- See the point specified in ‘Weaknesss’
- Is multi image necessary? Wouldn’t it be sufficient to train the model on more text-image compositional examples like those given in the MSS benchmark?

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
3

### Summary
The paper identifies the bottlenecks in current safety fine-tuning methods, the helpful harmless trade-off, and their failures on challenging safety tasks. To address these issues, the authors introduce MIS, a dataset designed for reasoning-based safety understanding across multiple images, and propose MIRage, a fine-tuning method that enhances safety reasoning without harming general utility.

### Strengths
1. The paper systematically diagnoses and analyzes current safety fine-tuning bottlenecks through comprehensive experiments.
2. Authors present MIS, the first multi-image safety dataset, featuring a training split aimed at enhancing models’ safety-related visual perception and reasoning abilities.
3. Models finetuned with MIRage generalize well to previously unseen safety categories.

### Weaknesses
1. The “Safety CoT” template is interesting but not well formalized or analyzed—it’s unclear how much of the improvement comes from reasoning versus simply from data diversity. The authors should include ablations to isolate the impact of Safety CoT vs. multi-image input vs. fine-tuning data scale.
2. While MIS and MIRage are presented as key contributions, the fine-tuning pipeline itself largely follows a standard SFT paradigm. There is limited method innovation in model architecture or training strategy, which somewhat reduces the technical novelty of the work.
3. The MIS dataset appears unbalanced across categories, with nearly half of the samples belonging to Illegal Activity, while categories such as Self-Harm, Privacy, and Erotic account for less than 7%.

### Questions
1. How is the “Safety CoT” annotation structured—does the model generate step-by-step reasoning before the final refusal, or is it a single concatenated response?
2. How will the number of general QA samples used in MIRage affect models’ general performance? 
3. The results in Table 4 show very high reasoning scores (RSR ~100%). Is it because after finetuning, the model has a very strong reasoning ability? Then what is the finetuned models’ performance on general multi-image reasoning datasets (e.g., ScienceQA) compared with other baselines?
4. In Table 5, after fine-tuning with MIRage, the model’s accuracy on MSS-Safe decreases by about 12%. Could the authors analyze the reason behind this drop?
5. Will the full dataset and finetuned models be released?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses a safety gap in vision–language models (VLMs), where many safety-tuned VLMs identify content in harmful text or single-image inputs, but fail in combinations of two images with a neutral prompt. Following from this observation the authors make the (general) claim that current safety SFT methods (text-only safety SFT, multimodal SFT) do not actually teach (visual–visual–text) safety reasoning; either the models answer harmful composite queries, or they become over-conservative.
To this end, they introduce a dataset MIS (multi-image safety benchmark): 2,185 test items and ~4k SFT-style training items across 6 safety domains (illegal activity, violence, hate, self-harm, privacy, erotic), with three difficulty/style splits (easy, hard, real). Each example has two images + a detoxified/neutral instruction where, importantly, the unsafe intent only comes from the image–image relation.

The paper then SFTs several VLMs (e.g. InternVL2.5-8B) on these examples, where labels corresond to COTs describing the images conten and explain why the combination is unsafe. They evaluate the models against post-hoc or reconstructed VLGuard baselines on the same base model, showing large improvements on attack success rates (ASR), while performance on general multimodal benchmarks stays consistent.

### Strengths
- The paper addresses an interesting and important topic: safety gaps in multi-image VLMs, where it shows the benefits of COT reasoning to address these limitations.
- It introduces an interesting small dataset (MIS).
- The paper studies an interesting safety-loophole in in VLMs using two neutral images + text; in this scope it is well-framed, and building on existing work where neutral text and single images can still produce unsafe outputs.
- Overall, the experiments and evaluations are thorough (including training several VLMs) include several safety baselines, public safety datasets (FigStep, MSSBench, SIUO), general benchmarks to check utility, including ablations.
- The paper shows that multi-modal benchmark performance does not drop as it shows it is the case in other safety methods.
- The training method is simple: standard SFT on the MIS dataset.

### Weaknesses
The paper shows several weaknesses hampering generalizability of the approach aimed at addressing "a significant bottleneck in the safety capabilities of existing safeguarding methods" [line 045]:

- The focus is solely on two images + text (appears limited to generalize this claim)
- Along this line the datasets size also appears limited, especially for the smaller categories (e.g. self-harm, privacy), with only a few hundred images.
- The method is only evaluated on internally trained models and not compared against existing safety-tuned VLMs, such as [1,2].
- Given the somewhat narrow focus, I find the presentation too strong at times, e.g. "existing SFT methods are insufficient to provide effective defenses.” [line 100], “reveal a significant bottleneck in the safety capabilities of existing safeguarding methods.”, “the cause of safety bottlenecks can primarily be attributed to (i) composition of SFT inputs and (ii) construction method of SFT labels.” [line 101]


[1] https://arxiv.org/abs/2406.05113
[2] https://arxiv.org/abs/2406.12030

### Questions
Can you run MIS on at least one released safety VLM?
How does the method perform when we use more than two images?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper diagnoses a key failure in existing safety fine-tuning for Vision-Language Models: methods either fail to stop harmful outputs or become "over-conservative," refusing to answer benign questions. The authors attribute this to a "safety visual reasoning gap," where models can't reason about harmful intent implied by (even safe) visual contexts. To solve this, they introduce the Multi-Image Safety (MIS) dataset, a novel, synthetically-generated dataset featuring multi-image inputs paired with safety Chain-of-Thought labels. Fine-tuning on this dataset (MIRage) is shown to effectively reduce attack success rates on their new benchmark while improving performance on general VLM tasks.

### Strengths
1. The MIS dataset is the first to focus on multi-image safety scenarios, a necessary and complex domain. The proposed MIRage, which uses safety CoT labels to teach reasoning rather than just refusal, is an effective solution to the problem. The dataset can be a useful resource to the community.
2. The method achieves a good balance of helpfulness and harmlessness. The results in Table 4 (near-0% ASR on their benchmark) and Table 6 (a slight increase in average accuracy on general benchmarks) show a good trade-off between safety and helpfulness.
3. The paper is well-written and easy to understand.

### Weaknesses
1. In Finding 1 and Discussion 1, the authors state that “fine-tuning models on such data leads to over-prudence on visual features, causing the model to reject benign visual inputs.” However, the exact experimental configuration for VLGuard-P is unclear. In the original VLGuard post-hoc fine-tuning setup, 5k general helpfulness samples were included. Therefore, two concerns arise: (1) if this experiment did not follow the same configuration, the conclusion may not be directly comparable; or (2) if it did follow the exact setup, then the total training set would include 6k benign image-text pairs—three times more than the 2k unsafe pairs—making “over-prudence on visual features” an unconvincing explanation.
2. Both the MIS training data and the primary test data (MIS-easy/hard) are generated using a T2I model (Stable Diffusion 3.5). And many of the benchmarks are synthetic such as FigStep, MM-Safety. It's unclear how it generalizes to real-image safety benchmarks. The "MIS-real" split, which would validate generalization, is too small to be conclusive.
3. The paper focuses exclusively on multimodal safety, providing zero evaluation on standard text-only safety benchmarks. A primary VLM attack method is still text-only jailbreaking, and it's unknown if this specialized visual safety fine-tuning has inadvertently created new vulnerabilities or "catastrophically forgotten" textual safety alignments.

### Questions
In Table 10, the inference latencies such as tokens/s change. Since the model is only fine-tuned on the MIRage dataset without a modification to the architecture, why would that change?

### Soundness
2

### Presentation
3

### Contribution
3
