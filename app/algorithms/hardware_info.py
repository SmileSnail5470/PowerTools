import re
import struct
import ctypes
import platform
import functools
import sys


_X86_64_MACHINES = frozenset({"x86_64", "amd64", "x64"})
_VNNI_FLAGS = frozenset({"avx512vnni", "avxvnni"})

# Win64 调用约定的机器码：void cpuid(uint32 leaf, uint32 subleaf, uint32 out[4])
_WIN64_CPUID_STUB = bytes((
    0x53,                    # push rbx
    0x89, 0xC8,              # mov  eax, ecx      ; leaf
    0x89, 0xD1,              # mov  ecx, edx      ; subleaf
    0x0F, 0xA2,              # cpuid
    0x41, 0x89, 0x00,        # mov  [r8],    eax
    0x41, 0x89, 0x58, 0x04,  # mov  [r8+4],  ebx
    0x41, 0x89, 0x48, 0x08,  # mov  [r8+8],  ecx
    0x41, 0x89, 0x50, 0x0C,  # mov  [r8+12], edx
    0x5B,                    # pop  rbx
    0xC3,                    # ret
))
# uint64 xgetbv(uint32 xcr)，xcr 已在 ecx 中
_WIN64_XGETBV_STUB = bytes((
    0x0F, 0x01, 0xD0,        # xgetbv
    0x48, 0xC1, 0xE2, 0x20,  # shl rdx, 32
    0x48, 0x09, 0xD0,        # or  rax, rdx
    0xC3,                    # ret
))


def _normalize_cpu_flags(flags) -> frozenset[str]:
    return frozenset(re.sub(r"[^a-z0-9]", "", flag.lower()) for flag in flags if flag)


def _windows_cpu_flags() -> frozenset[str]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32)
    kernel32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)

    code = _WIN64_CPUID_STUB + _WIN64_XGETBV_STUB
    mem = kernel32.VirtualAlloc(None, len(code), 0x3000, 0x40)  # MEM_COMMIT|MEM_RESERVE, PAGE_EXECUTE_READWRITE
    if not mem:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        ctypes.memmove(mem, code, len(code))
        cpuid_fn = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)(mem)
        xgetbv_fn = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32)(mem + len(_WIN64_CPUID_STUB))
        regs = (ctypes.c_uint32 * 4)()

        def cpuid(leaf: int, subleaf: int = 0) -> tuple[int, int, int, int]:
            cpuid_fn(leaf, subleaf, ctypes.addressof(regs))
            return tuple(regs)

        def bit(value: int, n: int) -> bool:
            return bool((value >> n) & 1)

        if cpuid(0)[0] < 7:
            return frozenset()
        ecx1 = cpuid(1)[2]
        if not (bit(ecx1, 27) and bit(ecx1, 28)):  # OSXSAVE + AVX
            return frozenset()
        xcr0 = xgetbv_fn(0)
        if xcr0 & 0x06 != 0x06:  # OS 未启用 XMM/YMM 状态，AVX 系列均不可用
            return frozenset()

        flags = set()
        eax7, ebx7, ecx7, _ = cpuid(7, 0)
        if bit(ebx7, 5):
            flags.add("avx2")
        if eax7 >= 1 and bit(cpuid(7, 1)[0], 4):
            flags.add("avxvnni")
        if xcr0 & 0xE0 == 0xE0:  # OS 已启用 opmask/ZMM 状态
            if bit(ebx7, 16):
                flags.add("avx512f")
            if bit(ecx7, 11):
                flags.add("avx512vnni")
        return frozenset(flags)
    finally:
        kernel32.VirtualFree(mem, 0, 0x8000)  # MEM_RELEASE


def _numpy_cpu_flags() -> frozenset[str]:
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:
        from numpy.core._multiarray_umath import __cpu_features__ as features
    return _normalize_cpu_flags(name for name, enabled in features.items() if enabled)


@functools.lru_cache(maxsize=1)
def _x86_cpu_flags() -> frozenset[str]:
    native_probe = {
        "win32": _windows_cpu_flags,
    }.get(sys.platform)
    for probe in filter(None, (native_probe, _numpy_cpu_flags)):
        try:
            flags = probe()
        except Exception as exc:
            continue
        if flags:
            return flags
    return frozenset()


@functools.lru_cache(maxsize=1)
def _set_x64quantprecision() -> bool:
    """是否开启 ORT 的 session.x64quantprecision（U8S8 -> U8U8）。

    x64 SSE4.1/AVX2/AVX512(无 VNNI) 的 U8S8 量化 GEMM 用 vpmaddubsw 做 int16 饱和累加，
    可能溢出；该选项改用较慢但精确的 U8U8，且仅在 AVX2/AVX512 平台生效。
    - 非 x86-64（ARM、Apple Silicon 等）：不适用。
    - 仅 SSE4.1：选项无效，不开启。
    - AVX2 / AVX512 且无 VNNI（Haswell~Skylake-X、Zen1~Zen3）：开启。
    - 有 AVX512-VNNI 或 AVX-VNNI（Cascade Lake+、Alder Lake+、Zen4+）：无溢出，不开启以保速度。
    探测失败时返回 False，与 ORT 默认行为一致。
    """
    if platform.machine().lower() not in _X86_64_MACHINES or struct.calcsize("P") != 8 or sys.platform != "win32":
        return False
    flags = _x86_cpu_flags()
    # 所有 AVX512 CPU 都支持 AVX2，判断 avx2 即可同时覆盖两者。
    has_avx2 = "avx2" in flags
    has_vnni = not flags.isdisjoint(_VNNI_FLAGS)
    enable = has_avx2 and not has_vnni
    return enable