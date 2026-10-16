"""
Unit tests for rulest.common: the hashcat rule validator, the pure-Python
rule applicator, FNV-1a hashing, and small helper utilities.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rulest import state
from rulest.common import (
    HashcatRuleValidator,
    should_exclude_rule,
    safe_worker_count,
    fnv1a_32,
    py_apply_chain,
    _py_apply_single_rule,
)


# ---------------------------------------------------------------------------
# HashcatRuleValidator.validate_rule_for_gpu
# ---------------------------------------------------------------------------

class TestValidateRuleForGpu:

    @pytest.mark.parametrize("rule", [
        ':',        # no-op
        'l', 'u', 'c', 'C', 't', 'r', 'd', 'f',    # single-char, no operand
        'q', 'k', 'K', 'E', '{', '}', '[', ']',
        'T0', 'D5', 'LA', 'RZ', '+3', '-9', '.1', ',2', "'0",  # position + char
        'i05', 'o1a', '30X',                                  # digit + digit/char
        'x01', '*05', 'O12',                                  # digit + digit
        'sab',                                                # substitute a->b
        '@x', 'e ', '$!', '^#',                               # char operand
        'z3', 'Z9', 'y2', 'Y1',                               # count + position-ish
        'T0 l u',                                             # chained rule (spaces)
    ])
    def test_valid_rules_accepted(self, rule):
        assert HashcatRuleValidator.validate_rule_for_gpu(rule) is True

    @pytest.mark.parametrize("rule", [
        'T',         # missing required position operand
        'D',
        'L',
        's',         # 's' needs two operands
        'sa',        # 's' needs two operands
        'i0',        # 'i' needs digit + char
        'x0',        # 'x' needs two digits
        '@',         # needs operand
        'Q',         # rejected single-char rule
        '!x',        # rejected two-char rule ('!' leading)
        'garbage!!', # unrecognized command
    ])
    def test_invalid_rules_rejected(self, rule):
        assert HashcatRuleValidator.validate_rule_for_gpu(rule) is False

    def test_rejects_rule_string_exceeding_max_gpu_rules(self):
        # MAX_GPU_RULES caps the number of ops; ':' is one op each.
        too_many = ':' * (HashcatRuleValidator.MAX_GPU_RULES + 1)
        assert HashcatRuleValidator.validate_rule_for_gpu(too_many) is False

    def test_accepts_rule_string_at_max_gpu_rules(self):
        exactly_max = ':' * HashcatRuleValidator.MAX_GPU_RULES
        assert HashcatRuleValidator.validate_rule_for_gpu(exactly_max) is True

    def test_empty_rule_string_is_valid_gpu_rule(self):
        # An empty rule parses to zero ops, which is <= MAX_GPU_RULES, so
        # the validator accepts it (it's a functional no-op, not malformed).
        assert HashcatRuleValidator.validate_rule_for_gpu('') is True

    def test_allow_reject_rules_flag_disables_exclusion(self):
        # 'Q' is normally excluded by should_exclude_rule.
        assert HashcatRuleValidator.validate_rule_for_gpu('Q') is False
        state.ALLOW_REJECT_RULES = True
        try:
            # cache is keyed on the string only, so use a rule not seen above
            assert should_exclude_rule('Q') is False
        finally:
            state.ALLOW_REJECT_RULES = False


class TestValidateRulesForGpu:

    def test_filters_out_invalid_rules_and_strips_newlines(self):
        rules = ['l\n', 'u\r\n', 'garbage!!', '', 'T0']
        result = HashcatRuleValidator.validate_rules_for_gpu(rules)
        assert result == ['l', 'u', 'T0']

    def test_empty_list_returns_empty_list(self):
        assert HashcatRuleValidator.validate_rules_for_gpu([]) == []


# ---------------------------------------------------------------------------
# should_exclude_rule
# ---------------------------------------------------------------------------

class TestShouldExcludeRule:

    @pytest.mark.parametrize("rule", ['_', 'M', '4', '6', 'Q'])
    def test_excludes_single_char_reject_rules(self, rule):
        assert should_exclude_rule(rule) is True

    @pytest.mark.parametrize("rule", ['!x', '/y', '(z', ')a', '<b', '>c', '_d'])
    def test_excludes_two_char_reject_rules(self, rule):
        assert should_exclude_rule(rule) is True

    @pytest.mark.parametrize("rule", ['?ab', '=cd', 'vef'])
    def test_excludes_three_char_reject_rules(self, rule):
        assert should_exclude_rule(rule) is True

    def test_does_not_exclude_ordinary_rules(self):
        assert should_exclude_rule('l') is False
        assert should_exclude_rule('T0') is False
        assert should_exclude_rule('') is False

    def test_empty_rule_never_excluded(self):
        assert should_exclude_rule('') is False


# ---------------------------------------------------------------------------
# safe_worker_count
# ---------------------------------------------------------------------------

class TestSafeWorkerCount:

    def test_clamps_to_at_least_one(self):
        assert safe_worker_count(0) == 1
        assert safe_worker_count(-5) == 1

    def test_passes_through_reasonable_values_on_posix(self):
        if os.name != 'nt':
            assert safe_worker_count(8) == 8

    def test_accepts_float_like_input(self):
        assert safe_worker_count(4.7) == 4


# ---------------------------------------------------------------------------
# fnv1a_32
# ---------------------------------------------------------------------------

class TestFnv1a32:

    def test_deterministic(self):
        assert fnv1a_32(b'hello') == fnv1a_32(b'hello')

    def test_different_inputs_generally_differ(self):
        assert fnv1a_32(b'hello') != fnv1a_32(b'world')

    def test_different_seeds_generally_differ(self):
        assert fnv1a_32(b'hello', seed=1) != fnv1a_32(b'hello', seed=2)

    def test_result_is_32_bit(self):
        h = fnv1a_32(b'some fairly long input string here')
        assert 0 <= h <= 0xFFFFFFFF

    def test_empty_bytes(self):
        h = fnv1a_32(b'')
        assert isinstance(h, int)


# ---------------------------------------------------------------------------
# py_apply_chain / _py_apply_single_rule (CPU rule applicator, used for
# GPU-result verification)
# ---------------------------------------------------------------------------

class TestApplySingleRule:

    def test_noop(self):
        assert _py_apply_single_rule(':', 'Hello') == 'Hello'

    def test_lowercase(self):
        assert _py_apply_single_rule('l', 'HeLLo') == 'hello'

    def test_uppercase(self):
        assert _py_apply_single_rule('u', 'HeLLo') == 'HELLO'

    def test_capitalize(self):
        assert _py_apply_single_rule('c', 'hello world') == 'Hello world'

    def test_invert_capitalize(self):
        assert _py_apply_single_rule('C', 'Hello') == 'hELLO'

    def test_toggle_case(self):
        assert _py_apply_single_rule('t', 'HeLLo') == 'hEllO'

    def test_reverse(self):
        assert _py_apply_single_rule('r', 'hello') == 'olleh'

    def test_duplicate(self):
        assert _py_apply_single_rule('d', 'ab') == 'abab'

    def test_reflect(self):
        assert _py_apply_single_rule('f', 'ab') == 'abba'

    def test_rotate_left(self):
        assert _py_apply_single_rule('{', 'abcd') == 'bcda'

    def test_rotate_right(self):
        assert _py_apply_single_rule('}', 'abcd') == 'dabc'

    def test_delete_first(self):
        assert _py_apply_single_rule('[', 'abcd') == 'bcd'

    def test_delete_last(self):
        assert _py_apply_single_rule(']', 'abcd') == 'abc'

    def test_prepend(self):
        assert _py_apply_single_rule('^!', 'abc') == '!abc'

    def test_append(self):
        assert _py_apply_single_rule('$!', 'abc') == 'abc!'

    def test_purge_char(self):
        assert _py_apply_single_rule('@a', 'banana') == 'bnn'

    def test_duplicate_all_n_times(self):
        assert _py_apply_single_rule('p2', 'ab') == 'ababab'

    def test_toggle_case_at_position(self):
        assert _py_apply_single_rule('T0', 'abc') == 'Abc'

    def test_delete_at_position(self):
        assert _py_apply_single_rule('D1', 'abc') == 'ac'

    def test_insert_at_position(self):
        assert _py_apply_single_rule('i1X', 'abc') == 'aXbc'

    def test_overwrite_at_position(self):
        assert _py_apply_single_rule('o1X', 'abc') == 'aXc'

    def test_truncate_at_position(self):
        assert _py_apply_single_rule("'2", 'abcdef') == 'ab'

    def test_extract_range(self):
        assert _py_apply_single_rule('x14', 'abcdefgh') == 'bcde'

    def test_omit_range(self):
        assert _py_apply_single_rule('O13', 'abcdefgh') == 'aefgh'

    def test_substitute_char(self):
        # 'sXY' replaces every X with Y.
        assert _py_apply_single_rule('sab', 'banana') == 'bbnbnb'

    def test_swap_first_two(self):
        assert _py_apply_single_rule('k', 'abcd') == 'bacd'

    def test_swap_last_two(self):
        assert _py_apply_single_rule('K', 'abcd') == 'abdc'

    def test_swap_at_positions(self):
        assert _py_apply_single_rule('*03', 'abcd') == 'dbca'

    def test_duplicate_first_n_and_prepend(self):
        assert _py_apply_single_rule('y2', 'abcdef') == 'ababcdef'

    def test_duplicate_last_n_and_append(self):
        assert _py_apply_single_rule('Y2', 'abcdef') == 'abcdefef'

    def test_duplicate_first_char_n_times_and_prepend(self):
        assert _py_apply_single_rule('z3', 'abc') == 'aaaabc'

    def test_duplicate_last_char_n_times_and_append(self):
        assert _py_apply_single_rule('Z3', 'abc') == 'abcccc'

    def test_bitwise_shift_left(self):
        # 'a' = 0x61, shifted left = 0xC2
        result = _py_apply_single_rule('L0', 'a')
        assert result == chr(0xC2)

    def test_ascii_increment_decrement(self):
        assert _py_apply_single_rule('+0', 'a') == 'b'
        assert _py_apply_single_rule('-0', 'b') == 'a'

    def test_unknown_rule_returns_none(self):
        assert _py_apply_single_rule('Q', 'abc') is None

    def test_empty_rule_is_noop(self):
        assert _py_apply_single_rule('', 'abc') == 'abc'

    def test_out_of_range_position_is_safe_noop_like(self):
        # Position beyond word length shouldn't raise or corrupt the word.
        assert _py_apply_single_rule('T9', 'ab') == 'ab'
        assert _py_apply_single_rule('D9', 'ab') == 'ab'


class TestApplyChain:

    def test_single_rule_chain(self):
        assert py_apply_chain('l', 'HELLO') == 'hello'

    def test_multi_rule_chain_applied_in_order(self):
        # capitalize then append '!'
        assert py_apply_chain('c $!', 'hello') == 'Hello!'

    def test_chain_with_unsupported_rule_returns_none(self):
        assert py_apply_chain('l Q u', 'HELLO') is None

    def test_empty_chain_returns_word_unchanged(self):
        assert py_apply_chain('', 'hello') == 'hello'

    def test_chain_order_matters(self):
        # capitalize-then-lowercase undoes the capitalization; the reverse
        # order does not, so chain order changes the result.
        assert py_apply_chain('c l', 'hello world') != py_apply_chain('l c', 'hello world')


if __name__ == "__main__":
    import pytest
    print(f"Running {__file__} directly — use `pytest {__file__}` for full output.\n")
    raise SystemExit(pytest.main([__file__, "-v"]))
