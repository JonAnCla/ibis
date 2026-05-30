"""GC-safety tests for Relation.fields caching.

These tests document the memory-retention behaviour of the
``_FieldCacheKey``-based WeakKeyDictionary cache that sits behind
``Relation.fields``.

Design summary
--------------
``Relation.fields`` is a cached property.  The cache is a module-level
``WeakKeyDictionary[_FieldCacheKey, FrozenOrderedDict[str, Column]]``.
Each ``Relation`` instance lazily allocates a ``_FieldCacheKey`` token and
stores it in a ``__slots__`` slot; this token — which holds *no* reference
back to the relation — becomes the dict key.

Why the CacheKey proxy alone does NOT eliminate retention
---------------------------------------------------------
The VALUE stored in the cache is a ``FrozenOrderedDict`` whose values are
``Field`` instances.  Every ``Field`` stores a *strong* reference to its
owning ``Relation`` in ``Field.rel`` (and in ``Field.__args__``, which the
``Concrete`` machinery uses for hashing and equality).  Because the cache
itself is a module-level global (a GC root), the reachability chain

    ``_relation_fields_cache``  →  ``FrozenOrderedDict``  →  ``Field``
    →  ``Field.rel``  →  ``rel``

is entirely composed of strong references, so ``rel`` is always reachable
from a root.  CPython's reference-counter therefore never drops ``rel``'s
count to zero, and the cycle collector cannot help either (``rel`` is not in
an *unreachable* cycle — it is reachable from the root).

The ``_FieldCacheKey`` mechanism does ensure that the weak-key entry is
evicted as soon as the *key* becomes unreachable (see
``test_cache_key_eviction_when_key_unreachable``), but the key is the
token, not the relation.  The relation is kept alive by the value, so the
token in the relation's slot is also kept alive, so the weak reference in the
dict never fires.

What would fix the retention
-----------------------------
Making the back-reference in the cache value *weak* would break the strong
chain and allow ``rel`` to be collected.  Two equivalent strategies:

  a. Store ``weakref.ref(rel)`` inside ``Field`` instead of ``rel`` itself
     (requires changes to ``Field.rel``, ``Field.__args__``, and all call
     sites that dereference ``field.rel``).

  b. Bypass ``Field`` objects in the cached value entirely — store only the
     schema names and build short-lived ``Field`` objects on each access.
     This is effectively the "no-cache" baseline and sacrifices the cache hit
     benefit.

Both options have non-trivial API impact and are tracked as separate work
items.  These tests exist to document the *current* behaviour, catch
regressions, and guide future work.
"""

from __future__ import annotations

import gc
import weakref

import pytest

import ibis
from ibis.expr.operations.relations import (
    Field,
    Relation,
    _FieldCacheKey,
    _relation_fields_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table(name: str = "t"):
    """Return a simple UnboundTable operation."""
    return ibis.table({"a": "int64", "b": "str"}, name=name).op()


# ---------------------------------------------------------------------------
# _FieldCacheKey unit tests
# ---------------------------------------------------------------------------


class TestFieldCacheKey:
    def test_is_lightweight(self):
        """_FieldCacheKey has no state — only __weakref__ in its slots."""
        key = _FieldCacheKey()
        assert not hasattr(key, "__dict__")
        assert key.__slots__ == ("__weakref__",)

    def test_identity_hash_and_equality(self):
        """Each token is its own identity; two tokens are never equal."""
        k1 = _FieldCacheKey()
        k2 = _FieldCacheKey()
        assert k1 is not k2
        assert k1 != k2
        assert hash(k1) != hash(k2) or id(k1) != id(k2)  # identity-based

    def test_weakly_referenceable(self):
        key = _FieldCacheKey()
        ref = weakref.ref(key)
        assert ref() is key
        del key
        gc.collect()
        assert ref() is None

    def test_cache_key_eviction_when_key_unreachable(self):
        """WeakKeyDict evicts the entry when the key token is collected."""
        from weakref import WeakKeyDictionary

        cache: WeakKeyDictionary = WeakKeyDictionary()
        key = _FieldCacheKey()
        cache[key] = "sentinel"
        assert len(cache) == 1

        del key
        gc.collect()
        assert len(cache) == 0  # entry was evicted


# ---------------------------------------------------------------------------
# Relation._field_cache_key slot
# ---------------------------------------------------------------------------


class TestRelationCacheKeySlot:
    def test_slot_absent_before_fields_access(self):
        op = _make_table()
        assert not hasattr(op, "_field_cache_key")

    def test_slot_populated_after_fields_access(self):
        op = _make_table()
        _ = op.fields
        assert hasattr(op, "_field_cache_key")
        assert isinstance(op._field_cache_key, _FieldCacheKey)

    def test_same_key_on_repeated_access(self):
        op = _make_table()
        _ = op.fields
        key1 = op._field_cache_key
        _ = op.fields
        key2 = op._field_cache_key
        assert key1 is key2

    def test_different_relations_get_different_keys(self):
        op1 = _make_table("t1")
        op2 = _make_table("t2")
        _ = op1.fields
        _ = op2.fields
        assert op1._field_cache_key is not op2._field_cache_key


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------


class TestFieldsCacheCorrectness:
    def test_fields_returns_correct_columns(self):
        op = _make_table()
        fields = op.fields
        assert set(fields.keys()) == {"a", "b"}

    def test_cache_hit_returns_same_object(self):
        op = _make_table()
        f1 = op.fields
        f2 = op.fields
        assert f1 is f2

    def test_field_values_are_field_instances(self):
        op = _make_table()
        for col in op.fields.values():
            assert isinstance(col, Field)

    def test_field_rel_points_to_owning_relation(self):
        op = _make_table()
        for field in op.fields.values():
            assert field.rel is op

    def test_cache_entry_absent_before_access(self):
        op = _make_table()
        assert not hasattr(op, "_field_cache_key")
        initial_size = len(_relation_fields_cache)
        _ = op.fields
        assert len(_relation_fields_cache) == initial_size + 1

    def test_cache_entry_present_after_access(self):
        op = _make_table()
        _ = op.fields
        assert op._field_cache_key in _relation_fields_cache


# ---------------------------------------------------------------------------
# GC / memory-retention tests
# ---------------------------------------------------------------------------


class TestGCRetention:
    """
    Document (and detect regressions in) the GC-retention behaviour.

    The tests in this class deliberately assert the *current* (imperfect)
    behaviour.  They serve as a baseline so that future improvements that
    eliminate retention can be verified by flipping the assertions.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN LIMITATION: Relation is retained even after all external "
            "references are dropped and gc.collect() is called.  "
            "Root cause: the cache VALUE (FrozenOrderedDict → Field → "
            "Field.rel → rel) holds a strong reference to rel, and the cache "
            "itself is a module-level global (root).  The CacheKey proxy only "
            "governs the *key* side of the WeakKeyDict; it cannot break the "
            "strong chain on the *value* side.  "
            "Fix: make Field.rel (and Field.__args__[0]) store a weakref "
            "instead of a strong reference."
        ),
    )
    def test_relation_collected_after_fields_access(self):
        """Relation should be GC-collectable after all external refs drop."""
        op = _make_table()
        _ = op.fields  # populate cache
        op_ref = weakref.ref(op)
        del op
        gc.collect()
        assert op_ref() is None, (
            "Relation was not collected — the fields cache is retaining it."
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "KNOWN LIMITATION: cache entry is not evicted when the relation "
            "is dropped, because the value holds the relation alive.  "
            "See test_relation_collected_after_fields_access for details."
        ),
    )
    def test_cache_entry_evicted_after_relation_dropped(self):
        """Cache entry should be removed when the owning relation is dropped."""
        op = _make_table()
        _ = op.fields
        key_ref = weakref.ref(op._field_cache_key)
        del op
        gc.collect()
        assert key_ref() is None, (
            "CacheKey is still alive — eviction did not occur."
        )
        assert key_ref() not in _relation_fields_cache

    def test_cache_eviction_mechanism_works_in_isolation(self):
        """
        Prove the WeakKeyDict eviction *mechanism* is correct in isolation.

        This test constructs a scenario where the value does NOT hold a
        strong back-reference to the key.  In that case, dropping the key
        causes immediate eviction — demonstrating that the mechanism itself
        is sound, and that the retention seen in the other tests is caused
        solely by the strong back-reference in the value.
        """
        import weakref as wr
        from weakref import WeakKeyDictionary

        cache: WeakKeyDictionary = WeakKeyDictionary()

        key = _FieldCacheKey()
        # Value holds NO strong reference to anything with a strong ref to key
        cache[key] = "no back-reference here"
        key_ref = wr.ref(key)

        del key
        gc.collect()

        assert key_ref() is None, "Key should have been collected"
        assert len(cache) == 0, "Cache entry should have been evicted"

    def test_strong_backref_in_value_prevents_eviction(self):
        """
        Prove that a strong back-reference in the value causes retention,
        regardless of whether the object or a proxy token is used as the key.
        """
        from weakref import WeakKeyDictionary

        class Owner:
            pass

        class ValueWithBackRef:
            def __init__(self, owner):
                self.owner = owner  # strong back-reference

        cache: WeakKeyDictionary = WeakKeyDictionary()

        owner = Owner()
        key = _FieldCacheKey()
        owner._key = key  # owner holds key strongly

        cache[key] = ValueWithBackRef(owner)  # value → owner (strong)
        owner_ref = weakref.ref(owner)

        del owner, key
        gc.collect()

        # owner is retained because:
        # cache (root) → value → value.owner → owner
        assert owner_ref() is not None, (
            "Owner should still be alive (retained via value back-reference)"
        )

        # Manually break the cycle to allow clean teardown
        del cache

    def test_weakref_in_value_allows_collection(self):
        """
        Prove that replacing the strong back-reference with a weakref in the
        value breaks the retention — owner IS collected.

        This is the behaviour we would get if ``Field.rel`` were a weakref.
        """
        import weakref as wr
        from weakref import WeakKeyDictionary

        class Owner:
            pass

        class ValueWithWeakRef:
            def __init__(self, owner):
                self._owner_ref = wr.ref(owner)  # WEAK back-reference

            @property
            def owner(self):
                return self._owner_ref()

        cache: WeakKeyDictionary = WeakKeyDictionary()

        owner = Owner()
        key = _FieldCacheKey()
        owner._key = key

        cache[key] = ValueWithWeakRef(owner)  # value → weakref(owner)
        owner_ref = wr.ref(owner)

        del owner, key
        gc.collect()

        assert owner_ref() is None, (
            "Owner should have been collected — weakref in value breaks the "
            "strong chain from root to owner."
        )
        assert len(cache) == 0, "Cache entry should have been evicted too"
