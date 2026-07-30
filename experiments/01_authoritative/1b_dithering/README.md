DAMA Validation: The ground truth dataset is validated against all six DAMA data quality dimensions (Accuracy, Completeness, Consistency, Timeliness, Validity, Uniqueness) as part of generation — the pipeline fails loudly if validation doesn't pass. The full audit result is saved to experiments_output/ground_truth/dama_audit_report.json as a verifiable artifact. To independently confirm the claim, re-run the validator yourself:

python shared/data_generation/validate_dama_dimensions.py \
    --input experiments_output/ground_truth/canonical_customers.json
