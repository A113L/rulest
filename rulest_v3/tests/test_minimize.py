"""Unit tests for rulest.minimize: signature-based functional minimization."""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest.minimize import compute_rule_signature_hash, minimize_by_signature


PROBES = ['abc', 'Hello', 'test123']


class TestComputeRuleSignatureHash:

    def test_deterministic_for_same_rule_and_probes(self):
        sig1 = compute_rule_signature_hash('l', PROBES)
        sig2 = compute_rule_signature_hash('l', PROBES)
        assert sig1 == sig2

    def test_functionally_equivalent_rules_share_signature(self):
        # 'u' twice in a row behaves like 'u' once on these probe words.
        sig_u = compute_rule_signature_hash('u', PROBES)
        sig_uu = compute_rule_signature_hash('u u', PROBES)
        assert sig_u == sig_uu

    def test_functionally_different_rules_have_different_signatures(self):
        sig_l = compute_rule_signature_hash('l', PROBES)
        sig_u = compute_rule_signature_hash('u', PROBES)
        assert sig_l != sig_u

    def test_unsupported_rule_returns_sentinel(self):
        sig = compute_rule_signature_hash('Q', PROBES)
        assert sig.startswith('__UNSUPPORTED__')
        assert sig == '__UNSUPPORTED__::Q'

    def test_signature_depends_on_probe_words(self):
        sig_a = compute_rule_signature_hash('c', ['abc'])
        sig_b = compute_rule_signature_hash('c', ['xyz'])
        assert sig_a != sig_b


class TestMinimizeBySignature:

    def test_empty_counter_returns_empty_counter(self):
        result = minimize_by_signature(Counter(), PROBES)
        assert result == Counter()

    def test_removes_functionally_duplicate_rules(self):
        # 'l' and 'l l' are functionally identical on these probes.
        counter = Counter({'l': 5, 'l l': 3})
        result = minimize_by_signature(counter, PROBES)
        assert len(result) == 1
        # The single survivor keeps the higher of the two counts.
        assert list(result.values())[0] == 5

    def test_keeps_functionally_distinct_rules(self):
        counter = Counter({'l': 5, 'u': 3, 'c': 1})
        result = minimize_by_signature(counter, PROBES)
        assert len(result) == 3
        assert set(result.keys()) == {'l', 'u', 'c'}

    def test_prefers_shorter_rule_on_tie(self):
        # Same count, functionally equivalent -> shorter chain should survive.
        counter = Counter({'u u u': 10, 'u': 10})
        result = minimize_by_signature(counter, PROBES)
        assert len(result) == 1
        assert list(result.keys())[0] == 'u'

    def test_unsupported_rules_each_keep_one_survivor(self):
        counter = Counter({'Q': 2, 'M': 4})
        result = minimize_by_signature(counter, PROBES)
        # Both are unsupported but distinct sentinel signatures (rule-specific),
        # so both should survive independently.
        assert set(result.keys()) == {'Q', 'M'}


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
