from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:
    Console = None
    Table = None


def _render_tabulate(report_dict, accuracy, f1_macro, f1_weighted):
    if tabulate is None:
        print("[WARN] 'tabulate' not installed. Falling back to plain text report.")
        print(classification_report_from_dict(report_dict))
        return

    table = []
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            table.append([
                str(label),
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1-score', 0.0):.4f}",
                metrics.get('support', 0)
            ])

    print("\n=== FINAL RESULTS (UNSEEN TEST SET) ===")
    print(tabulate(table, headers=["Class", "Precision", "Recall", "F1-score", "Support"], tablefmt="fancy_grid"))
    print(f"\nOverall Accuracy   : {accuracy:.4f}")
    print(f"Overall F1 Macro   : {f1_macro:.4f}")
    print(f"Overall F1 Weighted: {f1_weighted:.4f}")


def _render_rich(report_dict, accuracy, f1_macro, f1_weighted):
    if Console is None or Table is None:
        print("[WARN] 'rich' not installed. Falling back to plain text report.")
        print(classification_report_from_dict(report_dict))
        print(f"\nAccuracy   : {accuracy:.4f}")
        print(f"F1 Macro   : {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        return

    console = Console()
    table = Table(title="Classification Report", style="bold cyan")
    table.add_column("Class", justify="center")
    table.add_column("Precision", justify="center")
    table.add_column("Recall", justify="center")
    table.add_column("F1-score", justify="center")
    table.add_column("Support", justify="center")

    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            table.add_row(
                str(label),
                f"{metrics.get('precision', 0.0):.4f}",
                f"{metrics.get('recall', 0.0):.4f}",
                f"{metrics.get('f1-score', 0.0):.4f}",
                str(metrics.get('support', 0))
            )

    console.print("\n=== FINAL RESULTS (UNSEEN TEST SET) ===", style="bold green")
    console.print(table)
    console.print(f"[bold green]Accuracy[/]: {accuracy:.4f}")
    console.print(f"[bold yellow]F1 Macro[/]: {f1_macro:.4f}")
    console.print(f"[bold magenta]F1 Weighted[/]: {f1_weighted:.4f}")


def classification_report_from_dict(report_dict):
    lines = ["Class\tPrecision\tRecall\tF1-score\tSupport"]
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            lines.append(
                f"{label}\t{metrics.get('precision', 0.0):.4f}\t"
                f"{metrics.get('recall', 0.0):.4f}\t"
                f"{metrics.get('f1-score', 0.0):.4f}\t"
                f"{metrics.get('support', 0)}"
            )
    return "\n".join(lines)


def evaluate_on_test(model_pipeline, X_test, y_test, pretty: str = "tabulate"):

    predictions = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    f1_weighted = f1_score(y_test, predictions, average="weighted")

    report_dict = classification_report(y_test, predictions, output_dict=True)

    if pretty is None or pretty.lower() == "none":
        print("\n=== FINAL RESULTS (UNSEEN TEST SET) ===")
        print(classification_report(y_test, predictions))
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"F1 Macro   : {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
    elif pretty.lower() == "tabulate":
        _render_tabulate(report_dict, accuracy, f1_macro, f1_weighted)
    elif pretty.lower() == "rich":
        _render_rich(report_dict, accuracy, f1_macro, f1_weighted)
    else:
        print(f"[WARN] Unknown pretty mode '{pretty}'. Falling back to plain report.")
        print("\n=== FINAL RESULTS (UNSEEN TEST SET) ===")
        print(classification_report(y_test, predictions))
        print(f"Accuracy   : {accuracy:.4f}")
        print(f"F1 Macro   : {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report_dict
    }
    return metrics
