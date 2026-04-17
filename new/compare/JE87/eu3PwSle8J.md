# Review

## Summary
This paper proposes a defense mechanism against prompt injection attacks by injecting hierarchy signals into the intermediate representation of each layer of the LLM.

## Soundness
3

## Presentation
3

## Contribution
2

## Strengths
1. The proposed method is simple and easy to follow.
2. The paper is well-organized and easy to read.
3. The empirical results are promising.

## Weaknesses
1. The novelty of this paper is limited. The idea of "Instruction Hierarchy" has been proposed in previous works, and this paper simply changes the way of injecting the hierarchy signal. The proposed method is similar to the Instructional Segment Embedding (ISE) proposed in Wu et al., 2024, with the only difference being that the ISE embeds the signal into the token embedding while the proposed method embeds the signal into the intermediate representation. However, the paper does not discuss the difference between the two methods in detail, nor does it compare their effectiveness.
2. The paper does not provide a theoretical analysis of the proposed method. For example, the authors could provide a theoretical analysis of the separation ability of the proposed method.
3. The paper does not evaluate the proposed method on out-of-distribution (OOD) data, such as the PAP dataset proposed in Chen et al., 2024a. This is important because the attacker could use the OOD data to bypass the defense. The paper only evaluates the method on the Alpaca dataset, which may not be sufficient to demonstrate its generalizability.
4. The paper does not evaluate the proposed method against adaptive attacks. The attacker could adapt the attack to the proposed defense, such as using the inverse of the hierarchy signal to guide the attack. The paper only evaluates the method against existing attacks, which may not be sufficient to demonstrate its robustness.

## Questions
1. What is the key difference between the proposed method and the ISE method?
2. Could you provide a theoretical analysis of the proposed method?
3. Could you provide an evaluation of the proposed method on OOD data?
4. Could you provide an evaluation of the proposed method against adaptive attacks?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
4