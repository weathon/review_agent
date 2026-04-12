Do NOT do these things in your code:
- Production safe: code should NOT be production safe but debug friendly
- Fallback Logic: Code should NOT have fallback logic like return empty string when something failed. Error and retry is fine.
- Catching error without showing a message
- Abstraction and reusable: Code should NOT be abstract or reuseable unless it is actually needed (no single call abstraction, function less than 5 lines should not be abstracted)
