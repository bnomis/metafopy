"""Case generation: every case the audit runs is derived here from R's own description of metafor (the universe) and the two tables of
``data/``, never from a list somebody remembered to write. What cannot be derived is reported as uncovered, and a test fails on it.
"""

import dataclasses
import fnmatch
import typing

import audit.cases
import audit.domains
import audit.universe

Json = dict[str, typing.Any]

NO_PRINT = frozenset({'print'})
SKIP_GENERICS = frozenset({'print'})


@dataclasses.dataclass
class Generated:
    cases: list[audit.cases.Case]
    uncovered: list[str]
    unaccounted: list[str]


def default_string(default: str | None) -> str | None:
    if default is None:
        return None
    text = default.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"' and '"' not in text[1:-1]:
        return text[1:-1]
    return None


def option_values(
    universe: audit.universe.Universe,
    domains: audit.domains.Domains,
    function: str,
    entry: Json,
    formal: Json,
) -> tuple[list[typing.Any], str]:
    """The values to try for one argument and why there are none ('' when there are some, else 'skip' or 'uncovered')."""
    name: str = formal['name']
    if domains.skipped(function, name):
        return [], 'skip'
    default = formal.get('default')
    current = default_string(default)
    values: list[typing.Any] = []
    for choice in list(formal.get('choices') or []) + list(entry.get('body_choices', {}).get(name, [])):
        if choice != current and choice not in values:
            values.append(choice)
    if default in ('TRUE', 'FALSE'):
        values.append(default == 'FALSE')
    for v in domains.values(function, name) or []:
        if v not in values:
            values.append(v)
    return (values, '') if values else ([], 'uncovered')


def args_of(entry: Json, graphics: list[str], extra: list[str] | None = None) -> list[Json]:
    """The formals of a function plus the arguments it reads from ``...``, as one list of formal-like entries."""
    names = {f['name'] for f in entry['formals']}
    formals = [f for f in entry['formals'] if f['name'] != '...']
    formals += [{'name': d, 'default': None, 'choices': None} for d in entry.get('dots', []) if d not in names]
    if entry.get('draws') and '...' in names:
        formals += [{'name': g, 'default': None, 'choices': None} for g in [*graphics, *(extra or [])] if g not in {f['name'] for f in formals}]
    return formals


def caller(fn: str, target: Json | None, pre: list[Json] | None = None) -> typing.Callable[[Json], Json]:
    """The call spec of ``fn`` on a set of arguments: on ``target`` when given, after the ``pre`` calls when given."""

    def make(args: Json) -> Json:
        spec: Json = {'fn': fn, 'args': args}
        if target is not None:
            spec['target'] = target
        if pre:
            spec['pre'] = pre
        return spec

    return make


def templated(template: Json) -> typing.Callable[[Json], Json]:
    def make(args: Json) -> Json:
        return {**template, 'args': args}

    return make


def option_cases(
    domains: audit.domains.Domains,
    prefix: str,
    function: str,
    entry: Json,
    make_call: typing.Callable[[Json], Json],
    base: Json,
    needs: list[str],
    print_it: bool,
    meta: dict[str, str],
    gen: Generated,
) -> None:
    """R functions that read a setting with ``getOption`` (``na.action``): one case per value of the setting."""
    for name in entry.get('options_read', []):
        if name in domains.options_skip:
            continue
        values = domains.options.get(name)
        if not values:
            gen.uncovered.append(f'{function}(option {name})')
            continue
        for value in values:
            spec = make_call(base)
            spec['options'] = {name: value}
            gen.cases.append(
                audit.cases.Case(
                    id=f'{prefix}/option:{name}={audit.domains.label(value)}',
                    axis=prefix.split('/')[0],
                    call=spec,
                    needs=needs,
                    print=print_it,
                    meta={**meta, 'arg': f'option:{name}'},
                )
            )


def sweep(
    universe: audit.universe.Universe,
    domains: audit.domains.Domains,
    prefix: str,
    function: str,
    entry: Json,
    make_call: typing.Callable[[Json], Json],
    base: Json,
    needs: list[str],
    first_arg: str | None,
    print_it: bool,
    meta: dict[str, str],
    gen: Generated,
    when: Json | None = None,
) -> None:
    """One case per value of each argument; ``when`` has, per swept argument, base arguments it needs changed (``fmtt(call=TRUE)`` wants scalars)."""
    formals = args_of(entry, domains.graphics, domains.per_function.get(function.split('.')[0], {}).get('graphics'))
    known = {f['name'] for f in formals}
    for formal in formals:
        name: str = formal['name']
        if name == first_arg:
            continue
        values, why = option_values(universe, domains, function, entry, formal)
        if why == 'uncovered' and name in base:
            continue
        if why == 'uncovered':
            gen.uncovered.append(f'{function}({name})')
            continue
        for value in values:
            companions = {k: v for k, v in {**domains.companions.get(name, {}), **(when or {}).get(name, {})}.items() if k in known}
            args = {**base, **companions, name: value}
            gen.cases.append(
                audit.cases.Case(
                    id=f'{prefix}/{name}={audit.domains.label(value)}',
                    axis=prefix.split('/')[0],
                    call=make_call(args),
                    needs=sorted({*needs, *audit.domains.refs_in(args)}),
                    print=print_it,
                    meta={**meta, 'arg': name},
                )
            )
    option_cases(domains, prefix, function, entry, make_call, base, needs, print_it, meta, gen)


def generate(
    universe: audit.universe.Universe,
    domains: audit.domains.Domains,
    recipes: audit.domains.Recipes,
    fixtures: list[Json],
) -> Generated:
    gen = Generated(cases=[], uncovered=[], unaccounted=[])
    subjects = [fx for fx in fixtures if universe.classes_of(fx['name'])]
    fits = [fx for fx in subjects if 'rma' in universe.classes_of(fx['name']) and not fx.get('meta', {}).get('derived')]
    results = [fx for fx in subjects if fx not in fits]
    generics = [g for g in universe.generics() if g not in SKIP_GENERICS]
    fit_functions = [f for f in recipes.fit_functions if f in universe.functions]

    for fx in subjects:
        name = fx['name']
        classes = universe.classes_of(name)
        gen.cases.append(
            audit.cases.Case(
                id=f'matrix/print/{name}',
                axis='matrix',
                call={'fn': 'identity', 'args': {'x': {'$ref': name}}},
                needs=[name],
                print=True,
                meta={'fn': 'print', 'fit': name, 'class': classes[0]},
            )
        )
        for generic in generics:
            method = universe.method_for(generic, classes)
            if method is None or (fx not in fits and method['class'].startswith('rma')):
                continue
            draws = universe.draws(generic)
            pre = recipes.pre(generic, name)
            for k, args in enumerate(recipes.generic_calls(generic, name)):
                gen.cases.append(
                    audit.cases.Case(
                        id=f'matrix/{generic}/{name}' + (f'#{k}' if k else ''),
                        axis='matrix',
                        call={'fn': generic, 'target': {'$ref': name}, 'args': args, **({'pre': pre} if pre else {})},
                        needs=sorted({name, *audit.domains.refs_in(args), *audit.domains.refs_in(pre)}),
                        print=not draws,
                        meta={'fn': generic, 'fit': name, 'class': classes[0]},
                    )
                )
    for fx in fits:
        fit = fx['name']
        classes = universe.classes_of(fit)
        for fn in fit_functions:
            args = recipes.base_args(fn, fit)
            gen.cases.append(
                audit.cases.Case(
                    id=f'matrix/{fn}/{fit}',
                    axis='matrix',
                    call={'fn': fn, 'target': {'$ref': fit}, 'args': args},
                    needs=sorted({fit, *audit.domains.refs_in(args)}),
                    print=not universe.functions[fn].get('draws'),
                    meta={'fn': fn, 'fit': fit, 'class': classes[0]},
                )
            )
    del results

    for fx in subjects:
        if not fx.get('meta', {}).get('sweep'):
            continue
        fit = fx['name']
        classes = universe.classes_of(fit)
        for generic in [*generics, 'print']:
            method = universe.method_for(generic, classes)
            if method is None or universe.refuses(generic, classes) or (fx not in fits and method['class'].startswith('rma')):
                continue
            base = recipes.base_args(generic, fit)
            pre = recipes.pre(generic, fit)
            sweep(
                universe,
                domains,
                f'options/{generic}/{fit}',
                method['method'],
                method,
                caller(generic, {'$ref': fit}, pre),
                base,
                sorted({fit, *audit.domains.refs_in(pre)}),
                method['formals'][0]['name'],
                not universe.draws(generic),
                {'fn': generic, 'fit': fit, 'class': classes[0]},
                gen,
            )
        for fn in fit_functions if fx in fits else []:
            entry = universe.functions[fn]
            sweep(
                universe,
                domains,
                f'options/{fn}/{fit}',
                fn,
                entry,
                caller(fn, {'$ref': fit}),
                recipes.base_args(fn, fit),
                [fit],
                entry['formals'][0]['name'],
                not entry.get('draws'),
                {'fn': fn, 'fit': fit, 'class': classes[0]},
                gen,
            )

    for fx in fixtures:
        if fx.get('op'):
            continue
        template = {k: v for k, v in fx.items() if k not in ('name', 'meta')}
        gen.cases.append(
            audit.cases.Case(
                id=f'construct/{fx["name"]}',
                axis='construct',
                call=template,
                needs=audit.domains.refs_in(fx),
                print=True,
                meta={'fn': fx['fn'], 'fit': fx['name'], 'class': ''},
            )
        )
    for fx in fixtures:
        if not fx.get('meta', {}).get('sweep') or fx.get('op'):
            continue
        construct_entry = universe.functions.get(fx['fn'])
        if construct_entry is None:
            continue
        template = {k: v for k, v in fx.items() if k not in ('name', 'meta', 'args')}
        sweep(
            universe,
            domains,
            f'construct/{fx["name"]}',
            fx['fn'],
            construct_entry,
            templated(template),
            fx.get('args', {}),
            audit.domains.refs_in(fx),
            construct_entry['formals'][0]['name'] if fx.get('target') else None,
            True,
            {'fn': fx['fn'], 'fit': fx['name'], 'class': ''},
            gen,
        )

    standalone = dict(recipes.standalone())
    for pattern, spec in recipes.patterns().items():
        for fn, entry in universe.functions.items():
            if fnmatch.fnmatch(fn, pattern) and fn not in standalone:
                names = {f['name'] for f in entry['formals']}
                standalone[fn] = {'base': {k: v for k, v in spec.get('args', {}).items() if k in names}}
    for label, spec in standalone.items():
        fn = spec.get('fn', label)
        if fn not in universe.functions:
            gen.unaccounted.append(f'recipe for unknown function {fn}')
            continue
        entry = universe.functions[fn]
        base = spec.get('base', {})
        needs = sorted(set(audit.domains.refs_in(base)))
        gen.cases.append(audit.cases.Case(id=f'standalone/{label}', axis='standalone', call={'fn': fn, 'args': base}, needs=needs, print=True, meta={'fn': fn}))
        sweep(
            universe,
            domains,
            f'standalone/{label}',
            fn,
            entry,
            caller(fn, None),
            base,
            needs,
            None,
            True,
            {'fn': fn},
            gen,
            spec.get('when'),
        )

    for label, spec in recipes.emmeans().items():
        call = spec['call']
        gen.cases.append(
            audit.cases.Case(
                id=f'emmeans/{label}', axis='emmeans', call=call, needs=sorted(set(audit.domains.refs_in(call))), print=True, meta={'fn': call['fn']}
            )
        )

    for name in universe.datasets:
        gen.cases.append(audit.cases.Case(id=f'datasets/{name}', axis='datasets', call={'fn': 'identity', 'args': {'x': {'$data': name}}}, meta={'fn': name}))

    gen.unaccounted.extend(unaccounted_functions(universe, recipes, fixtures, universe.generics(), fit_functions))
    gen.uncovered = sorted(set(gen.uncovered))
    return gen


def unaccounted_functions(
    universe: audit.universe.Universe,
    recipes: audit.domains.Recipes,
    fixtures: list[Json],
    generics: list[str],
    fit_functions: list[str],
) -> list[str]:
    """Exported functions no case reaches and no recipe waives: a new or forgotten function of metafor, which must never go unnoticed."""
    constructors = {fx['fn'] for fx in fixtures if 'fn' in fx}
    waived = recipes.unprobed()
    out = []
    for name in universe.functions:
        stem = next((g for g in generics if name == g or name.startswith(f'{g}.')), None)
        if stem is not None or name in fit_functions or name in {spec.get('fn', label) for label, spec in recipes.standalone().items()} or name in waived:
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in recipes.patterns()):
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in waived):
            continue
        if name in constructors or name.rsplit('.', 1)[0] in constructors:
            continue
        out.append(f'function {name} has no recipe and is not waived')
    return out
