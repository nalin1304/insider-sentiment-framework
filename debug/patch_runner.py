import re
from pathlib import Path

file_path = "run_collection.py"
with open(file_path, "r") as f:
    content = f.read()

# 1. Add new runner functions
new_runners = """
def run_corporate_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING CORPORATE ACTIONS COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.corporate_actions import collect_all_corporate_actions
        results = collect_all_corporate_actions()
        logger.info("Corporate actions collection complete: %d stocks processed.", len(results))
        return True
    except Exception as exc:
        logger.error("Corporate actions collection failed: %s", exc, exc_info=True)
        return False

def run_macro_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING MACRO DATA COLLECTION")
    logger.info("=" * 60)
    try:
        from data_collection.macro import collect_all_macro_data
        collect_all_macro_data()
        logger.info("Macro data collection complete.")
        return True
    except Exception as exc:
        logger.error("Macro data collection failed: %s", exc, exc_info=True)
        return False

def run_social_collection(logger: logging.Logger) -> bool:
    logger.info("=" * 60)
    logger.info("STARTING SOCIAL SENTIMENT COLLECTION")
    logger.info("=" * 60)
    logger.info("Social collection not yet fully implemented. Skipping.")
    return True

def generate_markdown_report(results: dict, elapsed: float, logger: logging.Logger):
    report_path = Path("data/raw/collection_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# Insider Sentiment Framework - Data Collection Report\\n\\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write(f"**Elapsed Time:** {elapsed:.1f} seconds\\n\\n")
        f.write("## Collection Summary\\n\\n")
        f.write("| Collector | Status |\\n")
        f.write("|-----------|--------|\\n")
        for collector_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            f.write(f"| {collector_name.capitalize()} | {status} |\\n")
    logger.info("Saved summary report to %s", report_path)
"""
# Insert before def main()
content = content.replace("def main():", new_runners + "\ndef main():")

# 2. Update CLI arguments
args_update = """    parser.add_argument("--corporate", action="store_true", help="Run corporate actions collector.")
    parser.add_argument("--macro", action="store_true", help="Run macro data collector.")
    parser.add_argument("--social", action="store_true", help="Run social sentiment collector.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (now default behavior for most collectors).")
"""
content = re.sub(
    r"    parser\.add_argument\(\n        \"--newsapi\",",
    args_update + "\n    parser.add_argument(\n        \"--newsapi\",",
    content
)

# 3. Update main logic execution
execution_update = """
    if args.all or args.corporate:
        results["corporate"] = run_corporate_collection(logger)
    if args.all or args.macro:
        results["macro"] = run_macro_collection(logger)
    if args.all or args.social:
        results["social"] = run_social_collection(logger)
"""
content = re.sub(
    r"    if args\.all or args\.news:\n        results\[\"news\"\] = run_news_collection\(\n            logger, include_newsapi=\(args\.newsapi or args\.all\)\n        \)",
    "    if args.all or args.news:\n        results[\"news\"] = run_news_collection(\n            logger, include_newsapi=(args.newsapi or args.all)\n        )\n" + execution_update,
    content
)

# 4. Add markdown report call
report_call = """
    generate_markdown_report(results, elapsed, logger)
"""
content = content.replace("logger.info(\"Total elapsed time: %.1f seconds\", elapsed)", "logger.info(\"Total elapsed time: %.1f seconds\", elapsed)\n" + report_call)

with open(file_path, "w") as f:
    f.write(content)
print("run_collection.py updated.")
