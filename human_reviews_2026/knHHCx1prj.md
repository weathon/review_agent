# Recurrent Deep Differentiable Logic Gate Networks

- Decision: Reject
- Scores: 0, 0, 0

## Abstract
While differentiable logic gates have shown promise in feedforward networks, their application to sequential modeling remains unexplored. 
    This paper presents the first implementation of Recurrent Deep Differentiable Logic Gate Networks (RDDLGN), combining Boolean operations with recurrent architectures for sequence-to-sequence learning.
    Evaluated on WMT'14 English-German translation, RDDLGN achieves 5.00 BLEU and 30.9\% accuracy during training, approaching GRU performance (5.41 BLEU) and graceful degradation (4.39 BLEU) during inference. 
    This work establishes recurrent logic-based neural computation as viable, opening research directions for  FPGA acceleration in sequential modeling and other recursive network architectures.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
Deep Differentiable Logic Gate Networks (DDLGNs) are extended by recurrency, and then an encoder-decoder model is build based on that.

The model is tested on the WMT 2014 English-to-German translation task.

### Strengths
* Interesting novel architecture.
* Good motivation.
* Shifted monolingual prediction result is interesting.
* Some good ablations in the appendix.

### Weaknesses
* The experiments seems flawed. The paper reports about 5 BLEU on WMT’14 English-German translation. Usual numbers on this task are about 30-35 BLEU. So this is completely broken?
* Sequence lengths are way too short for reasonable realistic experiments.
* Models are way too small.
* Contradiction on vanishing gradients (see comments below).
* Contradiction on long-sequence handling (see comments below).
* Unclear parts.

### Questions
I don't understand the BLEU scores. Is that serious? Is that correct? Do you maybe measure sth different? They are far away from usual number that you would expect, which are in the range of 30-35 BLEU. They are so far off that this is either measured incorrectly, or measured sth differently, or totally broken. No matter what it is, it basically makes the whole work meaningless.

Why no cross attention, as it would be common for encoder-decoder models?

"The data is tokenized at the word level using regex-based splitting, with a shared
16,000-token vocabulary for English and German." - I don't understand what this means. Do you use BPE or SPM or sth like that?

The models are obviously way too small.

If the goal is an efficient model, you should show how it performs when you give it the same amount of compute as a normal-sized well-performing baseline (e.g. some reasonable-sized Transformer). Does it perform better then?

"sequence length of 16 tokens" - this just makes it a toy task. Increase it to at least 50-100 or so to make it actually interesting and relevant.

Contradiction on vanishing gradients:
Sec 5.5: "confirming the absence of vanishing or exploding gradients"
Sec 6: "Additionally, the architecture suffers from vanishing gradient problems"

Contradiction on long-sequence handling:
The shifted-token task is used to argue that RDDLGNs have superior long-range memorization capabilities (Fig 4).
The tokenizer ablation shows the model's performance collapses as sequence length increases (Table 2).

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper introduces Recurrent Deep Differentiable Logic Gate Networks (RDDLGN)—a recurrent seq2seq model whose layers are mixtures of relaxed Boolean operators and which can be “collapsed” after training into a purely logical (binary) network intended for efficient inference (e.g., on FPGAs). On WMT’14 En→De, the uncollapsed model reports 5.00 BLEU / 30.9% token accuracy, and the collapsed variant reports 4.39 BLEU / 27.7%, roughly between vanilla RNN and GRU baselines under the authors’ settings.

### Strengths
The paper extends DDLGNs to sequence modeling with an encoder–decoder built from logic layers.

The memory evaluation is interesting. The shifted-copy task shows RDDLGN maintains high accuracy at larger temporal offsets than RNN/GRU,

### Weaknesses
Baselines are quite small and non-standard for WMT14, 5-ish BLEU feel like garbage and not convincing it's actually a practical setup.

### Questions
I'd suggest the author why using such MT setup.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This review is written by the AC after reading the 2 reviews with 0 scores. Since there is one major, likely fatal issue raised by both the reviewers, it is appropriate to focus on that instead of an independent emergency reviewer. The issue is that the bleu score reported in this paper are around 5 (as opposed to 20-40), if the reviewer's interpretation is correct, this would makes any comparisons in the paper meaningless irrespective of what method was proposed, or what the exact setting was. The authors did highlight these bleu scores in the abstract, showing this result is key to the paper as opposed to a side experiment.

To provide some specific references, On the EN-DE newstest task, phrased based system from 2014 had ~20 bleu score:
https://web.archive.org/web/20140625052707/http://matrix.statmt.org/
 transformer reported 28 in 2017.
Table 25-26 of the "Findings of WMT 2014" https://aclanthology.org/W14-3302.pdf for the medical tasks also reported ~20 bleu scores.

~5 bleu score was reported in tables 2 and 3 of this submission, which is probably around a word dictionary look up.

The authors should respond to this in the response period.

### Strengths
not evaluated

### Weaknesses
not evaluated

### Questions
Can the authors explain why their range of BLEU scores are out of normal, see summary for specific pointers?

### Soundness
1

### Presentation
2

### Contribution
2
