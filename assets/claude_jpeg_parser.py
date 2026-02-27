"""
JPEG parser — static Python style.

Parses the marker/segment structure of a JPEG file and extracts:
  - Image dimensions and color components (SOF)
  - Quantization tables (DQT)
  - Huffman tables (DHT)
  - Restart interval (DRI)
  - Comments (COM)
  - APP0/JFIF metadata
  - Scan header (SOS)
"""

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Static-Python primitives
# ---------------------------------------------------------------------------

type u8  = int
type u16 = int
type u32 = int
type i32 = int

smt_pre  = __debug__
smt_post = __debug__

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ok[T]:
    value: T

@dataclass(frozen=True)
class Err[E]:
    error: E

type Result[T, E] = Ok[T] | Err[E]


def _err[T](msg: str) -> Result[T, str]:
    return Err(msg)


def _ok[T](val: T) -> Result[T, str]:
    return Ok(val)


# ---------------------------------------------------------------------------
# Marker constants
# ---------------------------------------------------------------------------

SOI: u16 = 0xFFD8
EOI: u16 = 0xFFD9
SOS: u16 = 0xFFDA
DQT: u16 = 0xFFDB
DNL: u16 = 0xFFDC
DRI: u16 = 0xFFDD
DHT: u16 = 0xFFC4
DAC: u16 = 0xFFCC
COM: u16 = 0xFFFE

SOF_MARKERS: dict[u16, str] = {
    0xFFC0: "Baseline DCT",
    0xFFC1: "Extended sequential DCT",
    0xFFC2: "Progressive DCT",
    0xFFC3: "Lossless",
    0xFFC5: "Differential sequential DCT",
    0xFFC6: "Differential progressive DCT",
    0xFFC7: "Differential lossless",
    0xFFC9: "Extended sequential DCT (arithmetic)",
    0xFFCA: "Progressive DCT (arithmetic)",
    0xFFCB: "Lossless (arithmetic)",
}

APP_MARKERS: dict[u16, str] = {
    0xFFE0: "APP0 (JFIF)",
    0xFFE1: "APP1 (EXIF/XMP)",
    0xFFE2: "APP2 (ICC Profile)",
    0xFFE3: "APP3", 0xFFE4: "APP4", 0xFFE5: "APP5",
    0xFFE6: "APP6", 0xFFE7: "APP7", 0xFFE8: "APP8",
    0xFFE9: "APP9", 0xFFEA: "APP10", 0xFFEB: "APP11",
    0xFFEC: "APP12 (Picture Info)",
    0xFFED: "APP13 (IPTC/Photoshop)",
    0xFFEE: "APP14 (Adobe)",
    0xFFEF: "APP15",
}

MARKER_NAMES: dict[u16, str] = {
    SOI: "SOI", EOI: "EOI", SOS: "SOS",
    DQT: "DQT", DNL: "DNL", DRI: "DRI",
    DHT: "DHT", DAC: "DAC", COM: "COM",
    **{m: n.split()[0] for m, n in SOF_MARKERS.items()},
    **APP_MARKERS,
}

_STANDALONE: frozenset[u16] = frozenset(
    [SOI, EOI] + list(range(0xFFD0, 0xFFD8))  # RST0–RST7
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    offset: u32
    marker: u16
    length: u16
    data: bytes

    @property
    def name(self) -> str:
        return MARKER_NAMES.get(self.marker, f"0x{self.marker:04X}")


@dataclass
class QuantizationTable:
    table_id: u8
    precision: u8        # 0 = 8-bit, 1 = 16-bit
    values: list[u16]    # 64 coefficients, zigzag order


@dataclass
class HuffmanTable:
    table_class: u8      # 0 = DC, 1 = AC
    table_id: u8
    counts: list[u8]     # counts[i] = number of codes of length i+1
    values: list[u8]


@dataclass
class Component:
    component_id: u8
    h_sampling: u8
    v_sampling: u8
    quant_table_id: u8


@dataclass
class FrameHeader:
    marker: u16
    frame_type: str
    precision: u8
    height: u16
    width: u16
    components: list[Component]


@dataclass
class ScanComponent:
    component_id: u8
    dc_table_id: u8
    ac_table_id: u8


@dataclass
class ScanHeader:
    components: list[ScanComponent]
    spectral_start: u8
    spectral_end: u8
    approx_high: u8
    approx_low: u8


@dataclass
class JFIFHeader:
    version_major: u8
    version_minor: u8
    density_units: u8
    x_density: u16
    y_density: u16
    thumbnail_width: u8
    thumbnail_height: u8


@dataclass
class JpegFile:
    path: str
    segments: list[Segment]            = field(default_factory=list)
    frame: FrameHeader | None          = None
    scan: ScanHeader | None            = None
    quant_tables: list[QuantizationTable] = field(default_factory=list)
    huffman_tables: list[HuffmanTable]    = field(default_factory=list)
    restart_interval: u16              = 0
    comments: list[str]                = field(default_factory=list)
    jfif: JFIFHeader | None            = None
    app_segments: list[tuple[str, bytes]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

@dataclass
class Reader:
    _data: bytes
    pos: u32 = 0

    @property
    def remaining(self) -> u32:
        return len(self._data) - self.pos

    def read(self, n: u32) -> Result[bytes, str]:
        if smt_pre:
            assert n >= 0

        chunk = self._data[self.pos : self.pos + n]
        if len(chunk) < n:
            return _err(
                f"unexpected EOF: wanted {n} bytes at offset {self.pos}, "
                f"got {len(chunk)}"
            )
        self.pos += n
        return _ok(chunk)

    def read_u8(self) -> Result[u8, str]:
        match self.read(1):
            case Err(error): return Err(error)
            case Ok(b): return _ok(b[0])

    def read_u16_be(self) -> Result[u16, str]:
        match self.read(2):
            case Err(error): return Err(error)
            case Ok(b): return _ok(struct.unpack(">H", b)[0])

    def peek(self, n: u32) -> bytes:
        return self._data[self.pos : self.pos + n]

    def seek(self, pos: u32) -> None:
        self.pos = pos


# ---------------------------------------------------------------------------
# Segment iterator
# ---------------------------------------------------------------------------

def _next_segment(r: Reader) -> Result[Segment | None, str]:
    """Read one segment from the current reader position. None signals EOF."""
    if r.remaining < 2:
        return _ok(None)

    offset = r.pos

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(b0):
            if b0 != 0xFF:
                return _err(f"expected 0xFF marker byte at offset {offset}, got {b0:#04x}")

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(marker_low):
            # Consume padding bytes
            while marker_low == 0xFF:
                match r.read_u8():
                    case Err(e): return Err(e)
                    case Ok(v): marker_low = v

    marker: u16 = (0xFF << 8) | marker_low

    if marker in _STANDALONE:
        return _ok(Segment(offset=offset, marker=marker, length=0, data=b""))

    match r.read_u16_be():
        case Err(e): return Err(e)
        case Ok(length):
            if length < 2:
                return _err(
                    f"invalid segment length {length} at offset {offset} "
                    f"for marker {marker:#06x}"
                )
            payload_len = length - 2

    match r.read(payload_len):
        case Err(e): return Err(e)
        case Ok(data): pass

    seg = Segment(offset=offset, marker=marker, length=length, data=data)

    if marker == SOS:
        match _skip_entropy_data(r):
            case Err(e): return Err(e)
            case Ok(_): pass

    return _ok(seg)


def _skip_entropy_data(r: Reader) -> Result[None, str]:
    """Skip entropy-coded scan data, respecting 0xFF 0x00 byte-stuffing."""
    while r.remaining >= 2:
        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(byte):
                if byte != 0xFF:
                    continue

        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(next_byte):
                match next_byte:
                    case 0x00:
                        continue                 # byte-stuffed, not a marker
                    case 0xFF:
                        while r.remaining and r.peek(1) == b"\xFF":
                            r.read_u8()          # consume padding
                        continue
                    case _:
                        r.pos -= 2              # real marker — back up
                        return _ok(None)

    return _ok(None)


# ---------------------------------------------------------------------------
# Segment parsers
# ---------------------------------------------------------------------------

def _parse_dqt(data: bytes) -> Result[list[QuantizationTable], str]:
    if smt_pre:
        assert len(data) > 0

    r = Reader(data)
    tables: list[QuantizationTable] = []

    while r.remaining >= 65:
        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(byte):
                precision: u8 = (byte >> 4) & 0xF
                table_id: u8  = byte & 0xF

        if precision == 0:
            match r.read(64):
                case Err(e): return Err(e)
                case Ok(raw): values: list[u16] = list(raw)
        else:
            values = []
            for _ in range(64):
                match r.read_u16_be():
                    case Err(e): return Err(e)
                    case Ok(v): values.append(v)

        tables.append(QuantizationTable(table_id, precision, values))

    return _ok(tables)


def _parse_dht(data: bytes) -> Result[list[HuffmanTable], str]:
    if smt_pre:
        assert len(data) >= 17

    r = Reader(data)
    tables: list[HuffmanTable] = []

    while r.remaining >= 17:
        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(byte):
                table_class: u8 = (byte >> 4) & 0xF
                table_id: u8    = byte & 0xF

        match r.read(16):
            case Err(e): return Err(e)
            case Ok(raw): counts: list[u8] = list(raw)

        total = sum(counts)
        match r.read(total):
            case Err(e): return Err(e)
            case Ok(raw): values: list[u8] = list(raw)

        tables.append(HuffmanTable(table_class, table_id, counts, values))

    return _ok(tables)


def _parse_sof(marker: u16, data: bytes) -> Result[FrameHeader, str]:
    if smt_pre:
        assert len(data) >= 6

    if len(data) < 6:
        return _err("SOF segment too short")

    r = Reader(data)

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(precision): pass

    match r.read_u16_be():
        case Err(e): return Err(e)
        case Ok(height): pass

    match r.read_u16_be():
        case Err(e): return Err(e)
        case Ok(width): pass

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(n_comp): pass

    components: list[Component] = []
    for _ in range(n_comp):
        if r.remaining < 3:
            return _err("SOF component descriptor truncated")

        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(comp_id): pass

        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(sampling):
                h_samp: u8 = (sampling >> 4) & 0xF
                v_samp: u8 = sampling & 0xF

        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(qt_id): pass

        components.append(Component(comp_id, h_samp, v_samp, qt_id))

    frame = FrameHeader(
        marker=marker,
        frame_type=SOF_MARKERS.get(marker, f"SOF {marker:#06x}"),
        precision=precision,
        height=height,
        width=width,
        components=components,
    )

    if smt_post:
        assert frame.width > 0
        assert frame.height > 0
        assert len(frame.components) == n_comp

    return _ok(frame)


def _parse_sos(data: bytes) -> Result[ScanHeader, str]:
    if smt_pre:
        assert len(data) >= 1

    r = Reader(data)

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(n_comp): pass

    scan_comps: list[ScanComponent] = []
    for _ in range(n_comp):
        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(comp_id): pass

        match r.read_u8():
            case Err(e): return Err(e)
            case Ok(tbl):
                dc_id: u8 = (tbl >> 4) & 0xF
                ac_id: u8 = tbl & 0xF

        scan_comps.append(ScanComponent(comp_id, dc_id, ac_id))

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(ss): pass

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(se): pass

    match r.read_u8():
        case Err(e): return Err(e)
        case Ok(ah_al):
            ah: u8 = (ah_al >> 4) & 0xF
            al: u8 = ah_al & 0xF

    return _ok(ScanHeader(scan_comps, ss, se, ah, al))


def _parse_dri(data: bytes) -> Result[u16, str]:
    if smt_pre:
        assert len(data) >= 2

    if len(data) < 2:
        return _err("DRI segment too short")
    return _ok(struct.unpack(">H", data[:2])[0])


def _parse_com(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _parse_app0(data: bytes) -> JFIFHeader | None:
    if len(data) < 14 or data[:5] != b"JFIF\x00":
        return None
    maj, minor, units, xd, yd, tw, th = struct.unpack(">BBBHHBB", data[5:14])
    return JFIFHeader(maj, minor, units, xd, yd, tw, th)


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def parse(path: str | Path) -> Result[JpegFile, str]:
    raw = Path(path).read_bytes()
    r = Reader(raw)
    jpeg = JpegFile(path=str(path))

    # Validate SOI
    match _next_segment(r):
        case Err(e): return Err(e)
        case Ok(seg):
            if seg is None or seg.marker != SOI:
                return _err(f"not a JPEG file: {path}")
            jpeg.segments.append(seg)

    while True:
        match _next_segment(r):
            case Err(e): return Err(e)
            case Ok(None): break
            case Ok(seg):
                jpeg.segments.append(seg)

                match seg.marker:
                    case m if m in SOF_MARKERS:
                        match _parse_sof(seg.marker, seg.data):
                            case Err(e): return Err(e)
                            case Ok(frame): jpeg.frame = frame

                    case 0xFFDB:  # DQT
                        match _parse_dqt(seg.data):
                            case Err(e): return Err(e)
                            case Ok(tables): jpeg.quant_tables.extend(tables)

                    case 0xFFC4:  # DHT
                        match _parse_dht(seg.data):
                            case Err(e): return Err(e)
                            case Ok(tables): jpeg.huffman_tables.extend(tables)

                    case 0xFFDA:  # SOS
                        match _parse_sos(seg.data):
                            case Err(e): return Err(e)
                            case Ok(scan): jpeg.scan = scan

                    case 0xFFDD:  # DRI
                        match _parse_dri(seg.data):
                            case Err(e): return Err(e)
                            case Ok(interval): jpeg.restart_interval = interval

                    case 0xFFFE:  # COM
                        jpeg.comments.append(_parse_com(seg.data))

                    case 0xFFE0:  # APP0
                        jpeg.jfif = _parse_app0(seg.data)
                        jpeg.app_segments.append((seg.name, seg.data))

                    case m if m in APP_MARKERS:
                        jpeg.app_segments.append((seg.name, seg.data))

                    case _:
                        pass

                if seg.marker == EOI:
                    break

    return _ok(jpeg)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_ZIGZAG = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
]


def _dequantize(values: list[u16]) -> list[u16]:
    out: list[u16] = [0] * 64
    for zigzag_pos, natural_pos in enumerate(_ZIGZAG):
        out[natural_pos] = values[zigzag_pos]
    return out


def report(jpeg: JpegFile) -> None:
    print(f"File : {jpeg.path}")
    print(f"Segments ({len(jpeg.segments)}):")
    for seg in jpeg.segments:
        size = f"{seg.length} B" if seg.length else "(no payload)"
        print(
            f"  {seg.marker:#06x}  {seg.name:<24}  "
            f"offset={seg.offset:#010x}  {size}"
        )

    print()
    if jpeg.jfif:
        j = jpeg.jfif
        units_map = {0: "aspect ratio", 1: "DPI", 2: "DPCM"}
        units = units_map.get(j.density_units, "unknown")
        print(
            f"JFIF : v{j.version_major}.{j.version_minor:02d}  "
            f"{j.x_density}x{j.y_density} {units}  "
            f"thumb {j.thumbnail_width}x{j.thumbnail_height}"
        )

    if jpeg.frame:
        f = jpeg.frame
        cs_map = {1: "Greyscale", 3: "YCbCr/RGB", 4: "CMYK"}
        cs = cs_map.get(len(f.components), "unknown")
        print(
            f"Frame: {f.frame_type}  {f.width}x{f.height}  "
            f"{f.precision}-bit  {cs}"
        )
        for c in f.components:
            print(
                f"       component {c.component_id}  "
                f"sampling {c.h_sampling}x{c.v_sampling}  "
                f"qtable {c.quant_table_id}"
            )

    if jpeg.restart_interval:
        print(f"DRI  : restart interval = {jpeg.restart_interval} MCUs")

    if jpeg.quant_tables:
        print(f"\nQuantization tables ({len(jpeg.quant_tables)}):")
        for qt in jpeg.quant_tables:
            prec = "8-bit" if qt.precision == 0 else "16-bit"
            print(f"  Table {qt.table_id} ({prec}):")
            for row in range(8):
                vals = _dequantize(qt.values)[row * 8 : row * 8 + 8]
                print("    " + "  ".join(f"{v:3d}" for v in vals))

    if jpeg.huffman_tables:
        print(f"\nHuffman tables ({len(jpeg.huffman_tables)}):")
        for ht in jpeg.huffman_tables:
            kind = "DC" if ht.table_class == 0 else "AC"
            total = sum(ht.counts)
            dist = {i + 1: c for i, c in enumerate(ht.counts) if c}
            print(f"  {kind} table {ht.table_id}: {total} codes")
            print("    " + "  ".join(f"L{l}={c}" for l, c in sorted(dist.items())))

    if jpeg.scan:
        s = jpeg.scan
        print(
            f"\nScan : {len(s.components)} component(s)  "
            f"spectral {s.spectral_start}-{s.spectral_end}  "
            f"approx {s.approx_high}/{s.approx_low}"
        )
        for sc in s.components:
            print(
                f"       component {sc.component_id}  "
                f"DC {sc.dc_table_id}  AC {sc.ac_table_id}"
            )

    if jpeg.comments:
        print(f"\nComments ({len(jpeg.comments)}):")
        for c in jpeg.comments:
            print(f"  {c!r}")

    if jpeg.app_segments:
        print(f"\nAPP segments ({len(jpeg.app_segments)}):")
        for name, data in jpeg.app_segments:
            sig = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data[:16])
            print(f"  {name}: {len(data)} B  [{sig}…]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> i32:
    args = sys.argv[1:]
    if not args:
        print("usage: jpeg_parser.py <file.jpg> [...]", file=sys.stderr)
        return 1

    ok = True
    for path in args:
        match parse(path):
            case Err(e):
                print(f"error: {path}: {e}", file=sys.stderr)
                ok = False
            case Ok(jpeg):
                report(jpeg)
        print()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
