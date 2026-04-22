# TokenSwap: Backdoor Attack on the Compositional Understanding of Large Vision-Language Models

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Large vision-language models (LVLMs) have achieved impressive performance across a wide range of vision-language tasks, while they remain vulnerable to backdoor attacks. Existing backdoor attacks on LVLMs aim to force the victim model to generate a predefined target pattern, which is either inserted into or replaces the original content. We find that these fixed-pattern attacks are relatively easy to detect, because the attacked LVLM tends to memorize such frequent patterns in the training dataset, thereby exhibiting overconfidence on these targets given poisoned inputs. To address these limitations, we introduce TokenSwap, a more evasive and stealthy backdoor attack that focuses on the compositional understanding capabilities of LVLMs. Instead of enforcing a fixed targeted content, TokenSwap subtly disrupts the understanding of object relationships in text. Specifically, it causes the backdoored model to generate outputs that mention the correct objects in the image but misrepresent their relationships (i.e., bags-of-words behavior). During training, TokenSwap injects a visual trigger into selected samples and simultaneously swaps the grammatical roles of key tokens in the corresponding textual answers. However, the poisoned samples exhibit only subtle differences from the original ones, making it challenging for the model to learn the backdoor behavior. To address this, TokenSwap employs an adaptive token-weighted loss that explicitly emphasizes the learning of swapped tokens, such that the visual triggers and bags-of-words behavior are associated. Extensive experiments demonstrate that TokenSwap achieves high attack success rates while maintaining superior evasiveness and stealthiness across multiple benchmarks and various LVLM architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TokenSwap, a backdoor attack designed to compromise the compositional understanding of large vision-language models (LVLMs).
The method poisons fine-grained text tokens by swapping object-role pairs in captions, combined with an adaptive token-weighted loss that focuses on low-confidence regions during training.
Experiments on multiple LVLMs show that the attack causes systematic misinterpretation of subject–object relationships while remaining visually and semantically stealthy.

### Strengths
- The paper is well written.

- The paper identifies an original threat direction by attacking LVLMs’ compositional reasoning rather than generic recognition or captioning functions.

- The proposed TokenSwap mechanism is intuitive and effective, successfully inducing structured semantic errors that are difficult to detect through standard evaluations.

### Weaknesses
- The novelty of the technical contribution is somewhat limited, as the adaptive weighting resembles prior focal-style losses, and token swapping is conceptually similar to existing data-poisoning schemes.

- The paper lacks theoretical analysis explaining why swapping subject–object roles so effectively destabilizes LVLMs’ alignment between vision and language.

- There is no detailed ablation on key factors such as poison ratio, swap frequency, or weighting hyperparameters, leaving uncertainty about robustness.

- The attack detectability is insufficiently discussed. While stealthiness is claimed, there is no analysis under existing backdoor detection or fine-tuning-based defenses.

- The practical risk is not fully contextualized—how likely such poisoning is to occur in real LVLM training pipelines remains speculative.

### Questions
See Weaknesses.

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
This paper presents TokenSwap, a novel backdoor attack designed to target the compositional understanding of large vision–language models (LVLMs). Unlike existing fixed-target attacks that insert or replace text patterns, TokenSwap subtly manipulates subject–object relationships in captions: when a visual trigger is present, the attacked model outputs captions with swapped grammatical roles (“a grass is eating a horse”). To make the subtle backdoor learnable, the authors introduce an Adaptive Token-Weighted (ATW) loss, which dynamically up-weights low-confidence swapped tokens, reinforcing their association with the trigger. Experiments across multiple LVLMs (BLIP-2, InstructBLIP, LLaVA-7B/13B) and datasets (MS-COCO, Flickr8k/30k) validate the effectiveness of the attack.

### Strengths
1: Instead of aiming for fixed textual patterns, the paper shifts focus to compositional reasoning, a higher-level semantic capability. 

2: The ATW loss is simple yet effective, demonstrating strong intuition and practical impact.

### Weaknesses
1: The paper claims that the proposed attack is more stealthy than traditional baselines. However, the evaluation of stealthiness is insufficient. The presented min-k perplexity distribution does not provide convincing evidence. For instance, the paper employs GPT-4O-mini to automatically detect token swaps, which demonstrates that the attack is not stealthy to GPT-4O-mini.

2: At line 081, the paper states that “contrastively pre-trained VLMs whose visual encoders are commonly used in most LVLMs often exhibit bag-of-words behavior, i.e., they have poor understanding of object order and relations in text (Yuksekgonul et al., 2023).” However, this citation does not appropriately support the claim that LVLMs exhibit bag-of-words behavior. Yuksekgonul et al. (2023) specifically demonstrate that the text encoder of contrastively pre-trained VLMs (e.g., CLIP) behaves like a bag of words. In contrast, LVLMs do not use the CLIP text encoder but rather a pretrained LLM as the text generator. Therefore, the findings of Yuksekgonul et al. (2023) are not directly relevant to the claim made in the paper.

What is more plausible in this case is that the CLIP visual encoder interprets the image, while the LLM generates the textual explanation. In this paper, the visual trigger is used as an external signal to deliberately distort the order of text generation. This design raises doubts about the practical relevance and reliability of the potential applications discussed in the Introduction. To substantiate the claim that LVLMs inherently exhibit bag-of-words behavior, additional empirical evidence and analysis are needed.

Reference: 

Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In ICLR, 2023.

### Questions
Q1: see weaknesses 1. 

Q2: see weaknesses 2.

Q3: if possible, can you show results of more recent LVLMs like Qwen 2.5-VL?

### Soundness
2

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
4

### Summary
TokenSwap proposes a backdoor attack on LVLMs that targets compositional understanding (disrupts the understanding of object relationships in text) rather than inserting/replacing a fixed phrase. The attack poisons training by stamping a visual trigger on images and swapping the subject–object tokens in the paired answer, aiming to induce “bag-of-words” behavior at test time. To make this subtle, instance-dependent behavior learnable, the authors introduce an Adaptive Token-Weighted (ATW) loss that enforce larger weights for those low-confidence swapped tokens. Experiments on Flickr8k, Flickr30k and COCO with BLIP-2, InstructBLIP, and LLaVA-1.5 (7B/13B) show higher attack success rates (ASR) than baselines while retaining utility on clean inputs; cross-dataset tests and simple clean fine-tuning as a defense are also reported.

### Strengths
1. Problem framing: Attacking compositional relations (subject–object) is a timely and underexplored angle relative to fixed-target insert/replace attacks; the min-k perplexity argument for detectability is persuasive. 
2. ATW loss is simple, well-motivated (emphasize the rare, low-confidence swapped tokens), and easy to reproduce; the mathematical form is clear. 
3. Broad model coverage (BLIP-2, InstructBLIP, LLaVA-7B/13B), in-dataset and cross-dataset tests, and comparisons to several attack families show consistent ASR gains with minimal utility loss.

### Weaknesses
1.ASR for relation swaps is detected by GPT-4o-mini + human inspection, which may introduce bias/variance. It would be better to justify the evaluation. 
2. The main results focus on captioning datasets. Evaluating on more tasks, such as VQA, would make better real-world impact.

3. Defense analysis: Only clean fine-tuning is studied. It would be better to involve more advanced defenses.

### Questions
1. The trigger is still visually detectable. Have you tested TokenSwap against common input-purification defenses (e.g., blur/smoothing) that typically weaken visible triggers? If not, could you comment on how robust the attack is under such defenses?
2.During training, the model uses a swap-token mask to guide learning. At inference, this mask is not available. How does the model reliably decide which tokens to swap when the behavior is object-dependent? Can you provide evidence that swapping generalizes to unseen objects?
3.Can the method naturally extend beyond subject–object swaps to other relation types (e.g., spatial roles, verb roles, multi-token phrase swaps)? Any preliminary observations would help clarify generality.

### Soundness
3

### Presentation
3

### Contribution
3
