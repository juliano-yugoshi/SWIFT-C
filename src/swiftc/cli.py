"""Unified command line dispatcher for the original SWIFT-C components."""
from __future__ import annotations
import sys

CLUSTER_COMMANDS = ('synthetic', 'realworld', 'sanity', 'inspect', 'configure', 'plot', 'report')
SUBCOMMANDS = CLUSTER_COMMANDS + ('discrete', 'select', 'retrieval', 'methods')

def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ('-h', '--help', 'help'):
        print('SWIFT-C: Semantic Weak-label Integration Framework for Thematic Clustering')
        print('Usage: swiftc COMMAND [OPTIONS]')
        print('Commands:')
        print('  methods    Show method IDs grouped by SWIFT-C family')
        print('  synthetic  Run synthetic clustering scenarios')
        print('  realworld  Cluster a mapped CSV or ZIP dataset')
        print('  discrete   Evaluate discrete weak-label baselines')
        print('  select     Select real-world methods from synthetic results')
        print('  retrieval  Evaluate cluster-based recommendation and retrieval')
        print('  report     Summarize an existing clustering runs.csv')
        print('  plot       Plot completed clustering results')
        print('  inspect    Inspect CSV or ZIP columns')
        print('  configure  Generate a dataset column mapping')
        print('  sanity     Execute a short internal correctness check')
        print('Run swiftc COMMAND --help for options.')
        return 0
    command, tail = args[0], args[1:]
    if command == 'methods':
        from .registry import format_method_catalogue
        print(format_method_catalogue())
        return 0
    if command in CLUSTER_COMMANDS:
        from .clustering.cli import main as submain
        return submain([command] + tail)
    if command == 'discrete':
        from .discrete.cli import main as submain
        return submain(tail)
    if command == 'select':
        from .selection.cli import main as submain
        return submain(tail)
    if command == 'retrieval':
        from .retrieval.cli import main as submain
        return submain(tail)
    print(f'Unknown SWIFT-C command: {command}', file=sys.stderr)
    print('Run swiftc --help for commands.', file=sys.stderr)
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
