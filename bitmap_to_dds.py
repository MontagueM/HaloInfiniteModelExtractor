from dataclasses import dataclass, fields, field
import numpy as np
import gf
import binascii
from typing import List
import os


with open('dxgi.format') as f:
    DXGI_FORMAT = f.readlines()


@dataclass
class ImageHeader:
    TargetSize: np.uint32 = np.uint32(0)  # 0
    TextureFormat: np.uint32 = np.uint32(0)  # 4
    Field8: np.uint32 = np.uint32(0)  # 8
    FieldC:  np.uint32 = np.uint32(0)  # C
    Field10: np.uint32 = np.uint32(0)  # 10
    Field14: np.uint32 = np.uint32(0)  # 14
    Field18: np.uint32 = np.uint32(0)  # 18
    Field1C: np.uint32 = np.uint32(0)  # 1C
    Cafe: np.uint16 = np.uint16(0)  # 20  0xCAFE
    Width: np.uint16 = np.uint16(0)  # 22
    Height: np.uint16 = np.uint16(0)  # 24
    Field26: np.uint16 = np.uint16(0)
    TA: np.uint16 = np.uint16(0)  # 28
    Field2A: np.uint16 = np.uint16(0)
    Field2C: np.uint32 = np.uint32(0)
    Field30: np.uint32 = np.uint32(0)
    Field34: np.uint32 = np.uint32(0)
    Field38: np.uint32 = np.uint32(0)
    LargeTextureHash: np.uint32 = np.uint32(0)  # 3C


@dataclass
class DX10Header:
    MagicNumber: np.uint32 = np.uint32(0)
    dwSize: np.uint32 = np.uint32(0)
    dwFlags: np.uint32 = np.uint32(0)
    dwHeight: np.uint32 = np.uint32(0)
    dwWidth: np.uint32 = np.uint32(0)
    dwPitchOrLinearSize: np.uint32 = np.uint32(0)
    dwDepth: np.uint32 = np.uint32(0)
    dwMipMapCount: np.uint32 = np.uint32(0)
    dwReserved1: List[np.uint32] = field(default_factory=list)  # size 11, [11]
    dwPFSize: np.uint32 = np.uint32(0)
    dwPFFlags: np.uint32 = np.uint32(0)
    dwPFFourCC: np.uint32 = np.uint32(0)
    dwPFRGBBitCount: np.uint32 = np.uint32(0)
    dwPFRBitMask: np.uint32 = np.uint32(0)
    dwPFGBitMask: np.uint32 = np.uint32(0)
    dwPFBBitMask: np.uint32 = np.uint32(0)
    dwPFABitMask: np.uint32 = np.uint32(0)
    dwCaps: np.uint32 = np.uint32(0)
    dwCaps2: np.uint32 = np.uint32(0)
    dwCaps3: np.uint32 = np.uint32(0)
    dwCaps4: np.uint32 = np.uint32(0)
    dwReserved2: np.uint32 = np.uint32(0)
    dxgiFormat: np.uint32 = np.uint32(0)
    resourceDimension: np.uint32 = np.uint32(0)
    miscFlag: np.uint32 = np.uint32(0)
    arraySize: np.uint32 = np.uint32(0)
    miscFlags2: np.uint32 = np.uint32(0)


@dataclass
class DDSHeader:
    MagicNumber: np.uint32 = np.uint32(0)
    dwSize: np.uint32 = np.uint32(0)
    dwFlags: np.uint32 = np.uint32(0)
    dwHeight: np.uint32 = np.uint32(0)
    dwWidth: np.uint32 = np.uint32(0)
    dwPitchOrLinearSize: np.uint32 = np.uint32(0)
    dwDepth: np.uint32 = np.uint32(0)
    dwMipMapCount: np.uint32 = np.uint32(0)
    dwReserved1: List[np.uint32] = field(default_factory=list)  # size 11, [11]
    dwPFSize: np.uint32 = np.uint32(0)
    dwPFFlags: np.uint32 = np.uint32(0)
    dwPFFourCC: np.uint32 = np.uint32(0)
    dwPFRGBBitCount: np.uint32 = np.uint32(0)
    dwPFRBitMask: np.uint32 = np.uint32(0)
    dwPFGBitMask: np.uint32 = np.uint32(0)
    dwPFBBitMask: np.uint32 = np.uint32(0)
    dwPFABitMask: np.uint32 = np.uint32(0)
    dwCaps: np.uint32 = np.uint32(0)
    dwCaps2: np.uint32 = np.uint32(0)
    dwCaps3: np.uint32 = np.uint32(0)
    dwCaps4: np.uint32 = np.uint32(0)
    dwReserved2: np.uint32 = np.uint32(0)


class Texture:
    def __init__(self):
        self.width = -1
        self.height = -1
        self.texture_format = -1
        self.array_size = 1
        self.fb = b""


def convert_bitmap(bitmap_handle, q):
    tex = Texture()
    fb = open(bitmap_handle, "rb").read()
    if len(fb) < 0x310:
        print("Texture has no data")
        return
    tex.width = gf.get_uint16(fb, len(fb)-4)
    tex.height = gf.get_uint16(fb, len(fb)-2)
    tex.texture_format = fb[0x309]
    # tex.texture_format = 0x48
    if "{pc}" not in bitmap_handle:
        raise Exception("Need to be pc")
    # form = bitmap_handle.split("{pc}")[0].split("_")[-:]
    # print(form)
    # Finding largest
    d = {x.split('_bitmap_resource_handle')[-1][-2]: x for x in os.listdir(folder) if ".chunk" in x and q in x}
    if len(d.keys()) == 0:
        print(f"{bitmap_handle} has no texture files to use, skipping...")
        return
    chunk_to_use = d[max(d.keys())]
    tex.fb = open(folder + chunk_to_use, "rb").read()
    print(chunk_to_use)
    save_direc = f"{out_path}/{'/'.join(folder.split('/')[:-1]).replace(unpack_directory, '')}"
    os.makedirs(save_direc, exist_ok=True)
    # if "normal" in bitmap_handle:
    #     tex.texture_format = 0x54
    # if "control" in bitmap_handle and "asg" not in bitmap_handle and "mask" not in bitmap_handle:
    #     tex.texture_format = 0x61
    write_texture(tex, f"{save_direc}/{q}.dds")


def write_texture(tex, full_save_path):

    form = DXGI_FORMAT[tex.texture_format]
    if '_BC' in form:
        dds_header = DX10Header()  # 0x0
    else:
        dds_header = DDSHeader()  # 0x0

    dds_header.MagicNumber = int('20534444', 16)  # 0x4
    dds_header.dwSize = 124  # 0x8
    dds_header.dwFlags = (0x1 + 0x2 + 0x4 + 0x1000) + 0x8
    dds_header.dwHeight = tex.height  # 0xC
    dds_header.dwWidth = tex.width  # 0x10
    dds_header.dwDepth = 0
    dds_header.dwMipMapCount = 0
    dds_header.dwReserved1 = [0]*11
    dds_header.dwPFSize = 32
    dds_header.dwPFRGBBitCount = 0
    dds_header.dwPFRGBBitCount = 32
    dds_header.dwPFRBitMask = 0xFF  # RGBA so FF first, but this is endian flipped
    dds_header.dwPFGBitMask = 0xFF00
    dds_header.dwPFBBitMask = 0xFF0000
    dds_header.dwPFABitMask = 0xFF000000
    dds_header.dwCaps = 0x1000
    dds_header.dwCaps2 = 0
    dds_header.dwCaps3 = 0
    dds_header.dwCaps4 = 0
    dds_header.dwReserved2 = 0
    if '_BC' in form:
        dds_header.dwPFFlags = 0x1 + 0x4  # contains alpha data + contains compressed RGB data
        dds_header.dwPFFourCC = int.from_bytes(b'\x44\x58\x31\x30', byteorder='little')
        dds_header.dxgiFormat = tex.texture_format
        dds_header.resourceDimension = 3  # DDS_DIMENSION_TEXTURE2D
        if tex.array_size % 6 == 0:
            # Compressed cubemap
            dds_header.miscFlag = 4
            dds_header.arraySize = int(tex.array_size / 6)
        else:
            # Compressed BCn
            dds_header.miscFlag = 0
            dds_header.arraySize = 1
    else:
        # Uncompressed
        dds_header.dwPFFlags = 0x1 + 0x40  # contains alpha data + contains uncompressed RGB data
        dds_header.dwPFFourCC = 0
        dds_header.miscFlag = 0
        dds_header.arraySize = 1
        dds_header.miscFlags2 = 0x1

    write_file(dds_header, tex, full_save_path)


def write_file(header, tex, full_save_path):
    with open(full_save_path, 'wb') as b:
        for f in fields(header):
            if f.type == np.uint32:
                flipped = "".join(gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(np.uint32(getattr(header, f.name)))[2:], 8), 8))
            elif f.type == List[np.uint32]:
                flipped = ''
                for val in getattr(header, f.name):
                    flipped += "".join(
                        gf.get_flipped_hex(gf.fill_hex_with_zeros(hex(np.uint32(val))[2:], 8), 8))
            else:
                print(f'ERROR {f.type}')
                return
            b.write(binascii.unhexlify(flipped))
        b.write(tex.fb)


def all_from_folder():
    extract = [x for x in os.listdir(folder) if x.endswith(".bitmap")]
    print(extract)
    for x in extract:
        convert_bitmap(folder + x, x)


def all_from_directory():
    global folder
    p = [os.path.join(dp, f)[len(directory):].replace("\\", "/") for dp, dn, fn in os.walk(os.path.expanduser(directory)) for f in fn if f.endswith(".bitmap") and ".chunk" not in f and ".dds" not in f ]
    for path in p:
        x = path.split('/')[-1]
        folder = directory + path.replace(x, "")
        convert_bitmap(directory + path, x)


if __name__ == "__main__":
    # if the code isnt working try replacing all the backslashes with forward slashes in every directory
    unpack_directory = "G:/HaloInfiniteUnpack"
    directory = f"{unpack_directory}/__chore\pc__\objects\weapons/"
    directory = directory.replace("\\", "/")
    out_path = "C:/Users/monta\OneDrive\ReverseEngineering\Halo\Extract/textures/"
    all_from_directory()