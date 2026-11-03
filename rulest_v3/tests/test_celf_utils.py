from rulest.celf_utils import target_ids_from_seen_words


def test_target_ids_from_seen_words_decodes_little_endian_bits():
    words = [
        (1 << 0) | (1 << 5) | (1 << 31),
        (1 << 0) | (1 << 7) | (1 << 30),
    ]
    assert target_ids_from_seen_words(words, 64) == {0, 5, 31, 32, 39, 62}


def test_target_ids_from_seen_words_ignores_padding_bits_and_masks_uint32():
    assert target_ids_from_seen_words([0xFFFFFFFF, 0xFFFFFFFF], 35) == set(range(35))
    assert target_ids_from_seen_words([0x1_0000_0001], 2) == {0}
    assert target_ids_from_seen_words([], 20) == set()
