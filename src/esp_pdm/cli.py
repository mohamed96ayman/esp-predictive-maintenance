"""Command-line entry points; training imports are deferred."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog="esp-pdm")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Prepare BASIC windows from SPE CSVs")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--out", default="data/processed")
    prepare.add_argument("--reference-stats", help="Replay a specified normalization artifact")
    prepare.add_argument("--max-gap", type=int, default=5)
    for name in ("evaluate", "train"):
        command = sub.add_parser(name)
        command.add_argument("--data", required=True)
        command.add_argument("--out", default="runs/evaluation" if name == "evaluate" else "models")
        command.add_argument("--threads", type=int, default=8)
        command.add_argument("--no-track", action="store_true")
        if name == "evaluate":
            command.add_argument("--max-folds", type=int)
            command.add_argument("--reference-oof")
    request = sub.add_parser("export-request")
    request.add_argument("--data", required=True)
    request.add_argument("--row", type=int, required=True)
    request.add_argument("--model-dir", default="models")
    request.add_argument("--out", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(args.model_dir), host=args.host, port=args.port)
        return
    if args.command == "prepare":
        from .preprocessing import prepare
        result = prepare(args.input, args.out, args.reference_stats, args.max_gap)
    else:
        from . import training
        if args.command == "evaluate":
            if args.max_folds is not None and args.max_folds < 1:
                parser.error("--max-folds must be positive")
            result = training.evaluate(args.data, args.out, not args.no_track, args.threads,
                                       args.max_folds, args.reference_oof)
        elif args.command == "train":
            result = training.train(args.data, args.out, not args.no_track, args.threads)
        else:
            result = training.export_request(args.data, args.row, args.model_dir, args.out)
    keys = ("role", "protocol", "n_folds", "pooled", "well_f1", "reference_comparison",
            "model_version", "preprocessing_id", "n_training_windows", "window_shape", "output")
    summary = {key: result[key] for key in keys if key in result}
    print(json.dumps(summary, indent=2))
