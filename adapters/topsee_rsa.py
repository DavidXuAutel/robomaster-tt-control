"""RSA PKCS#1 v1.5 公钥加密（纯标准库）。

拓普视登录流程要求用平台下发的公钥加密密码：
  GET /service/api/permission/free/security/rsa?securityKey=<k>  → base64 公钥
  POST /service/api/permission/free/pc/login                     → password 为加密后 base64

本模块只做公钥加密，不做解密/签名，因此不需要恒定时间实现。
之所以自己写而不引 cryptography/pycryptodome：`requirements.txt` 里没有加密库，
而项目约定不为单个功能新增第三方依赖。
"""

from __future__ import annotations

import base64
import os
from typing import Tuple


class RsaKeyError(ValueError):
    """公钥无法解析。"""


def _read_len(data: bytes, i: int) -> Tuple[int, int]:
    """读 DER 长度字段，返回 (长度, 下一个偏移)。"""
    if i >= len(data):
        raise RsaKeyError("DER 截断：缺少长度字段")
    first = data[i]
    i += 1
    if first < 0x80:
        return first, i
    n = first & 0x7F
    if n == 0 or n > 4 or i + n > len(data):
        raise RsaKeyError(f"DER 长度字段非法: 0x{first:02x}")
    return int.from_bytes(data[i : i + n], "big"), i + n


def _expect_tag(data: bytes, i: int, tag: int) -> Tuple[int, int]:
    """校验 tag 并读长度，返回 (内容长度, 内容起始偏移)。"""
    if i >= len(data) or data[i] != tag:
        got = f"0x{data[i]:02x}" if i < len(data) else "EOF"
        raise RsaKeyError(f"DER 期望 tag 0x{tag:02x}，实际 {got}")
    return _read_len(data, i + 1)


def _read_int(data: bytes, i: int) -> Tuple[int, int]:
    """读 DER INTEGER，返回 (值, 下一个偏移)。"""
    length, start = _expect_tag(data, i, 0x02)
    end = start + length
    if end > len(data):
        raise RsaKeyError("DER INTEGER 截断")
    return int.from_bytes(data[start:end], "big"), end


def parse_public_key(pem_or_b64: str) -> Tuple[int, int]:
    """解析公钥，返回 (modulus n, exponent e)。

    同时接受：
      - X.509 SubjectPublicKeyInfo（Java `getEncoded()` 的默认形态，平台最常见）
      - PKCS#1 RSAPublicKey（裸 SEQUENCE{n,e}）
      - 带或不带 PEM 头尾、含换行/空格的 base64
    失败抛 RsaKeyError。
    """
    body = "".join(
        line
        for line in pem_or_b64.strip().splitlines()
        if not line.startswith("-----")
    )
    body = "".join(body.split())
    if not body:
        raise RsaKeyError("公钥为空")
    try:
        der = base64.b64decode(body, validate=True)
    except Exception as exc:  # noqa: BLE001 — 统一成 RsaKeyError
        raise RsaKeyError(f"公钥不是合法 base64: {exc}") from exc

    # 外层必须是 SEQUENCE
    _outer_len, i = _expect_tag(der, 0, 0x30)

    # PKCS#1 形态：SEQUENCE { INTEGER n, INTEGER e }
    if i < len(der) and der[i] == 0x02:
        n, j = _read_int(der, i)
        e, _ = _read_int(der, j)
        return _check_key(n, e)

    # SPKI 形态：SEQUENCE { AlgorithmIdentifier, BIT STRING { PKCS#1 } }
    if i < len(der) and der[i] == 0x30:
        alg_len, alg_start = _expect_tag(der, i, 0x30)
        bit_len, bit_start = _expect_tag(der, alg_start + alg_len, 0x03)
        if bit_len < 1 or der[bit_start] != 0x00:
            raise RsaKeyError("SPKI BIT STRING 的 unused-bits 必须为 0")
        inner = der[bit_start + 1 : bit_start + bit_len]
        _inner_len, k = _expect_tag(inner, 0, 0x30)
        n, j = _read_int(inner, k)
        e, _ = _read_int(inner, j)
        return _check_key(n, e)

    raise RsaKeyError("无法识别的公钥结构（既非 PKCS#1 也非 SPKI）")


def _check_key(n: int, e: int) -> Tuple[int, int]:
    if n <= 0 or e <= 0:
        raise RsaKeyError("公钥参数必须为正")
    if n.bit_length() < 512:
        raise RsaKeyError(f"模数过短（{n.bit_length()} bit），拒绝使用")
    return n, e


def encrypt_b64(plaintext: str, pem_or_b64: str) -> str:
    """用公钥做 PKCS#1 v1.5 加密，返回 base64 密文。

    明文长度上限为 k-11 字节（k 为模数字节数），超出抛 ValueError。
    """
    n, e = parse_public_key(pem_or_b64)
    k = (n.bit_length() + 7) // 8
    msg = plaintext.encode("utf-8")
    if len(msg) > k - 11:
        raise ValueError(f"明文 {len(msg)} 字节超出 RSA 上限 {k - 11}")

    # EM = 0x00 || 0x02 || PS(非零随机, >=8字节) || 0x00 || M
    ps_len = k - len(msg) - 3
    ps = bytearray()
    while len(ps) < ps_len:
        chunk = os.urandom(ps_len - len(ps))
        ps.extend(b for b in chunk if b != 0)
    em = b"\x00\x02" + bytes(ps[:ps_len]) + b"\x00" + msg

    cipher = pow(int.from_bytes(em, "big"), e, n)
    return base64.b64encode(cipher.to_bytes(k, "big")).decode("ascii")
