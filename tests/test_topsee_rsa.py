"""M7a：纯标准库 RSA PKCS#1 v1.5 加密。

加密正确性靠假平台的私钥真解一遍来验证 —— 只测「不抛异常」等于没测。
"""

import base64

import pytest

from adapters.topsee_rsa import RsaKeyError, encrypt_b64, parse_public_key
from tests.fixtures.topsee_fake import (
    TEST_E,
    TEST_N,
    TEST_PKCS1_B64,
    TEST_SPKI_B64,
    rsa_decrypt,
)


def test_parse_spki():
    n, e = parse_public_key(TEST_SPKI_B64)
    assert (n, e) == (TEST_N, TEST_E)


def test_parse_pkcs1():
    n, e = parse_public_key(TEST_PKCS1_B64)
    assert (n, e) == (TEST_N, TEST_E)


def test_parse_pem_with_headers_and_newlines():
    pem = "-----BEGIN PUBLIC KEY-----\n"
    pem += "\n".join(TEST_SPKI_B64[i : i + 64] for i in range(0, len(TEST_SPKI_B64), 64))
    pem += "\n-----END PUBLIC KEY-----\n"
    assert parse_public_key(pem) == (TEST_N, TEST_E)


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "not-base64!!!", base64.b64encode(b"\x30\x03\x02\x01\x01").decode()],
)
def test_bad_key_rejected(bad):
    with pytest.raises(RsaKeyError):
        parse_public_key(bad)


def test_short_modulus_rejected():
    # SEQUENCE{INTEGER 0x0101(短模数), INTEGER 3} —— 结构合法但模数过短
    der = b"\x30\x08\x02\x03\x00\x01\x01\x02\x01\x03"
    with pytest.raises(RsaKeyError, match="模数过短"):
        parse_public_key(base64.b64encode(der).decode())


@pytest.mark.parametrize("plain", ["pa55w0rd!", "中文密码测试", "a", "x" * 100])
def test_encrypt_roundtrip(plain):
    assert rsa_decrypt(encrypt_b64(plain, TEST_SPKI_B64)) == plain


def test_encrypt_is_randomized():
    """PKCS#1 v1.5 每次填充随机，同一明文两次密文必须不同。"""
    a = encrypt_b64("same", TEST_SPKI_B64)
    b = encrypt_b64("same", TEST_SPKI_B64)
    assert a != b
    assert rsa_decrypt(a) == rsa_decrypt(b) == "same"


def test_plaintext_too_long_rejected():
    # 1024 位模数 → 上限 128-11 = 117 字节
    with pytest.raises(ValueError, match="超出 RSA 上限"):
        encrypt_b64("x" * 118, TEST_SPKI_B64)


def test_padding_has_no_zero_bytes():
    """PS 段不得含 0x00，否则解密方会提前截断。多跑几轮抓概率性 bug。"""
    for _ in range(50):
        assert rsa_decrypt(encrypt_b64("p", TEST_SPKI_B64)) == "p"
