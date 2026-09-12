"""Vỏ mỏng cho ``evaluation.cli`` — mọi quyết định nằm trong module đã có test.

    python scripts/run_eval.py --models baseline xlmr mbert visobert --full
"""

from evaluation.cli import main

if __name__ == "__main__":
    main()
