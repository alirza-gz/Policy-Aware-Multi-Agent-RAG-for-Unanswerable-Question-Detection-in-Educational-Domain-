"""Evaluation harness for the Policy-Aware Multi-Agent RAG system.

Modules:
    prepare_data : build the evaluation corpus and labelled question set.
    run_eval     : run the pipeline over the question set (governance ON/OFF).
    metrics      : compute quantitative metrics (accuracy, P/R/F1, FRR, abstain rate).
    analyze      : aggregate results, produce tables and Matplotlib figures.
"""
