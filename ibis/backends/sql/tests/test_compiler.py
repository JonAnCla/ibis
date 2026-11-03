from __future__ import annotations

import pytest
import sqlglot as sg

import ibis
from ibis import _
from ibis.backends import _get_backend_names
from ibis.backends.sql.dialects import Trino


def test_window_with_row_number_compiles():
    # GH #8058: the add_order_by_to_empty_ranking_window_functions rule was
    # matching on `RankBase` subclasses with a pattern expecting an `arg`
    # attribute, which is not present on `RowNumber`
    expr = (
        ibis.memtable({"a": range(30)})
        .mutate(id=ibis.row_number())
        .sample(fraction=0.25, seed=0)
        .mutate(is_test=_.id.isin(_.id))
        .filter(~_.is_test)
    )
    assert ibis.to_sql(expr)


def test_transpile_join():
    (result,) = sg.transpile(
        "SELECT * FROM t1 JOIN t2 ON x = y", read="duckdb", write=Trino
    )
    assert "CROSS JOIN" not in result


@pytest.mark.parametrize("dialect", sorted(_get_backend_names(exclude=("polars",))))
def test_to_sql_knows_every_sql_backend(dialect):
    # `ibis.to_sql` looks the dialect up on `ibis.backends.sql.compilers`, so
    # every SQL backend's compiler module has to be imported there
    expr = ibis.table({"a": "int64"}, name="t")
    assert ibis.to_sql(expr, dialect=dialect)


def test_select_merging_complexity_is_memoized(mocker):
    import sys

    import ibis.expr.operations as ops
    from ibis.backends.sql import rewrites

    def reference_complexity(node):
        def accum(node, *args):
            if isinstance(node, ops.Field):
                return 1
            elif isinstance(node, ops.Impure):
                return sys.maxsize
            else:
                return 1 + sum(args)

        return node.map_nodes(accum)[node]

    t = ibis.table({f"a{i}": "int64" for i in range(5)}, name="t")
    expr = t.mutate(r=ibis.random())
    for i in range(10):
        expr = expr.filter(expr.a0 > i).select("a0", "a1", "r", b=expr.a2 * 2)
        expr = expr.rename(a2="b").mutate(a3=_.a2 + 1, a4=0)

    # the memoized implementation computes the same scores
    cache = {}
    for node in expr.op().find(ops.Node):
        expected = reference_complexity(node)
        assert rewrites.complexity(node) == expected
        assert rewrites.complexity(node, cache) == expected

    # and a single cache is shared between all merge attempts of a compilation
    spy = mocker.spy(rewrites, "complexity")
    ibis.to_sql(expr, dialect="duckdb")
    caches = {id(call.args[1]) for call in spy.call_args_list}
    assert spy.call_count > 2
    assert len(caches) == 1
