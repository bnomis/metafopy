"""The audit command line: ``bin/audit.py`` runs every step in order and writes the short wrong-or-missing report."""

import argparse
import pathlib
import sys
import time

import audit.config
import audit.recall
import audit.report
import audit.steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='metafor-audit.py',
        description='Compare metafopy with R metafor: enumerate metafor, run every probe in both languages and report what is wrong or missing.',
    )
    parser.add_argument(
        '--out',
        type=pathlib.Path,
        default=None,
        help='Directory for the report and the raw results (default .audit/runs/<timestamp>)',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=4,
        help='Parallel R and Python processes',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=audit.config.CASE_TIMEOUT,
        help='Seconds a case may take in each language',
    )
    parser.add_argument(
        '--steps',
        default=None,
        help='Comma-separated steps to run instead of all of them',
    )
    parser.add_argument(
        '--only',
        action='append',
        default=None,
        help='Run only the cases whose id contains this text, or matches it when it has * or ? in it (repeatable)',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run a tenth of the cases (a smoke run, not an audit)',
    )
    parser.add_argument(
        '--metafopy-src',
        type=pathlib.Path,
        default=None,
        help='Directory that holds the metafopy package to test (default: this checkout); used by --recall to test older commits',
    )
    parser.add_argument(
        '--no-ledger',
        action='store_true',
        help='Report known differences as findings too',
    )
    parser.add_argument(
        '--recall',
        nargs='?',
        const='all',
        default=None,
        metavar='AUDITS',
        help='Instead of an audit, check that the harness reproduces the items of earlier audits (comma-separated numbers, default all)',
    )
    parser.add_argument(
        '--recall-full',
        action='store_true',
        help='With --recall: run every step in full on each commit instead of only the cases the items need',
    )
    parser.add_argument(
        '--update-universe',
        action='store_true',
        help='Write the R universe of this run to data/universe.json (after metafor or a fixture changed)',
    )
    parser.add_argument(
        '--rerender',
        type=pathlib.Path,
        default=None,
        metavar='RUN_DIR',
        help='Write the report of an earlier run again from its findings.json with the current known.toml (after triage), without running anything',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List the steps and exit',
    )
    return parser.parse_args(argv)


def recall(args: argparse.Namespace) -> int:
    corpus = audit.recall.Corpus.load()
    numbers = sorted(corpus.audits) if args.recall == 'all' else [int(n) for n in args.recall.split(',')]
    out = args.out or audit.config.WORK / 'runs' / ('recall-' + time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, bool]] = {}
    for number in numbers:
        items = [i for i in corpus.items if i.audit == number]
        findings = audit.recall.run_audit(number, corpus.audits[number], items, out, args.jobs, args.recall_full)
        results[number] = audit.recall.evaluate(items, findings)
        print(f'[recall] audit {number}: {sum(results[number].values())} of {len(items)} reproduced', flush=True)
    path, ok = audit.recall.write(out, corpus, results, {n: corpus.audits[n]['commit'] for n in numbers})
    print(f'recall: {path}')
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.recall is not None:
        return recall(args)
    if args.rerender is not None:
        print(f'report: {audit.report.rerender(args.rerender, args.no_ledger)}')
        return 0
    if args.list:
        for step in audit.steps.STEPS:
            print(f'{step.name:<12} {step.title}')
        return 0
    wanted = set(args.steps.split(',')) if args.steps else None
    unknown = (wanted or set()) - {s.name for s in audit.steps.STEPS}
    if unknown:
        print(f'unknown steps: {", ".join(sorted(unknown))}', file=sys.stderr)
        return 2
    out = args.out or audit.config.WORK / 'runs' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    ctx = audit.steps.Context(
        work=out, jobs=args.jobs, timeout=args.timeout, quick=args.quick, only=args.only, update_universe=args.update_universe, no_ledger=args.no_ledger
    )
    outcomes: dict[str, audit.steps.Outcome] = {}
    for step in audit.steps.STEPS:
        if wanted is not None and step.name not in wanted and step.name not in audit.steps.ALWAYS:
            continue
        started = time.monotonic()
        print(f'[{step.name}] {step.title}', flush=True)
        outcomes[step.name] = step.run(ctx)
        print(
            f'[{step.name}] {len(outcomes[step.name].findings)} findings, {"; ".join(outcomes[step.name].notes)} ({time.monotonic() - started:.0f}s)',
            flush=True,
        )
    path = audit.report.write(ctx, outcomes, out)
    print(f'report: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
