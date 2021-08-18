# This is the good version of the file

# Plug in a .render_model[0_mesh_resource]

import gf
import os
import pyfbx
import copy
import scipy.spatial


# folder_path = "H:/HIU/__chore/gen__/objects/weapons/pistol/needler/"
# folder_path = "H:\HIU\__chore\gen__\objects\weapons\melee\energy_sword/"
# folder_path = "H:/HIU/__chore/gen__/objects/weapons/pistol/sidearm_pistol/"
folder_path = "H:/HIU/objects/characters/spartan_armor/"

folder_path = folder_path.replace("\\", "/")
file = ""
for f in os.listdir(folder_path):
    if f[-30:] == ".render_model[0_mesh_resource]":
        file = folder_path + "/" + f
        break
if not file:
    raise Exception("No model in here")
file = file.replace("//", "/")
# folder_path = '/'.join(file.split('/')[:-1])
item_name = file.split('/')[-2]

fb = open(file, "rb").read()

# ## offset is above A0 first, that - 0x10
# data_offset = 0x690  # olympus_helmet_004_gear_001
# data_offset = 0x6E0  # needler
# data_offset = 0x8DB60  # spartan_armor
# # data_offset = 0xE684  # pelican
# data_offset = 0x4814 # assault rifle
# # data_offset = 0xE6D8  # scorpion
# # data_offset = 0x690  # olympus_torso_gear_006
# # data_offset = 0x65FCC  # marine

# Getting data offset
data_offset = fb.find(b"\x0A\x00\x00\x00\x08\x00\xBC\xBC")
if data_offset == -1:
    raise Exception("Could not find data offset")
data_offset -= 0x10

table_offset = 0x50 + gf.get_uint32(fb, 0x1C)*0x10 + gf.get_uint32(fb, 0x20)*0x20
table_count = gf.get_uint32(fb, 0x24)
potential_delta = data_offset - table_offset - table_count*0x14

model = pyfbx.Model()

## Getting automatic scale data and parts data
rmfile = file.replace("[0_mesh_resource]", "")
rmfb = open(rmfile, "rb").read()
rmfb_data_offset = gf.get_uint32(rmfb, 0x38)
# Finding render_model data
t1_offset = 0x50 + gf.get_uint32(rmfb, 0x18) * 0x18
t1_count = gf.get_uint32(rmfb, 0x1C)
t2_offset = t1_offset + t1_count*0x10
t2_count = gf.get_uint32(rmfb, 0x20)

t1es = []
t2es = []


class Table1Entry:
    def __init__(self):
        self.data_length = -1
        self.data_offset = -1


class Table2Entry:
    def __init__(self):
        self.t1e_reference = None
        self.t1e_parent = None
        self.hash = b''


for i in range(t1_offset, t1_offset+t1_count*0x10, 0x10):
    t1e = Table1Entry()
    t1e.data_length = gf.get_uint32(rmfb, i)
    t1e.data_offset = gf.get_uint32(rmfb, i+0x8) + rmfb_data_offset
    t1es.append(t1e)

for i in range(t2_offset, t2_offset+t2_count*0x20, 0x20):
    t2e = Table2Entry()
    t2e.hash = rmfb[i:i+0x10]
    ref = gf.get_int32(rmfb, i+0x14)
    if ref != -1:
        t2e.t1e_reference = t1es[ref]
    ref = gf.get_int32(rmfb, i+0x18)
    if ref != -1:
        t2e.t1e_parent = t1es[ref]
    t2es.append(t2e)

scale_data_offset = -1
part_offsets = []

for t2e in t2es:
    if t2e.hash == b"\xAC\xFD\x51\xFE\x78\x47\xFF\x62\x54\x30\xC3\xA8\x6C\xA9\x23\xA0":
        scale_data_offset = t2e.t1e_reference.data_offset + 4
    elif t2e.hash == b"\x9D\x84\x81\x4A\xB4\x42\xEE\xFB\xAC\x56\xC9\xA3\x18\x0F\x53\xE6":
        part_offsets.append(t2e)# - 6)

if scale_data_offset == -1:
    raise Exception("Could not find scale offset")
if not part_offsets:
    raise Exception("Could not find any parts")

# Strings
strings = {}
string_table_offset = t2_offset+t2_count*0x20
string_table_count = gf.get_uint32(rmfb, 0x28)
strings_offset = string_table_offset+string_table_count*0x10
for i in range(string_table_offset, string_table_offset+string_table_count*0x10, 0x10):
    # unk0x0 = gf.get_uint32(rmfb, i)
    # unk0x4 = gf.get_uint32(rmfb, i+4)
    string_offset = gf.get_uint32(rmfb, i+8)
    index = gf.get_int32(rmfb, i+0xC)
    if index == -1:
        break
    strings[index] = gf.offset_to_string_mem(rmfb, strings_offset+string_offset).replace('\\', '/')

# Reading scale data
scale_data = rmfb[scale_data_offset:scale_data_offset+80]
x_min = gf.get_float32(scale_data, 0x00)
x_max = gf.get_float32(scale_data, 0x04)
y_min = gf.get_float32(scale_data, 0x08)
y_max = gf.get_float32(scale_data, 0x0C)
z_min = gf.get_float32(scale_data, 0x10)
z_max = gf.get_float32(scale_data, 0x14)
u0_min = gf.get_float32(scale_data, 0x18)
u0_max = gf.get_float32(scale_data, 0x1C)
v0_min = gf.get_float32(scale_data, 0x20)
v0_max = gf.get_float32(scale_data, 0x24)
u1_min = gf.get_float32(scale_data, 0x18)
u1_max = gf.get_float32(scale_data, 0x1C)
v1_min = gf.get_float32(scale_data, 0x20)
v1_max = gf.get_float32(scale_data, 0x24)
model_scale = [[x_min, x_max, x_max-x_min], [y_min, y_max, y_max-y_min], [z_min, z_max, z_max-z_min]]
uv0_scale = [[u0_min, u0_max, u0_max-u0_min], [v0_min, v0_max, v0_max-v0_min]]
uv1_scale = [[u1_min, u1_max, u1_max-u1_min], [v1_min, v1_max, v1_max-v1_min]]
# More data here, dunno what its for though

# Processing parts
parts = []
class Part:
    def __init__(self):
        self.mat_index = -1
        self.index_offset = -1
        self.index_count = -1
        self.vertex_count = -1
        self.mat_string = ""


for p in part_offsets:
    offset = p.t1e_reference.data_offset
    length = p.t1e_reference.data_length
    parts.append([])
    for o in range(offset, offset+length, 0x18):
        part = Part()
        part.mat_index = gf.get_uint16(rmfb, o)
        part.index_offset = gf.get_uint32(rmfb, o+0x4)
        part.index_count = gf.get_uint32(rmfb, o+0x8)
        part.vertex_count = gf.get_uint16(rmfb, o+0x14)
        part.mat_string = strings[part.mat_index]
        parts[-1].append(part)
        a = 0

# todo delete testing parts stuff
max_mat_index = 0
for p in parts:
    for x in p:
        if x.mat_index > max_mat_index:
            max_mat_index = x.mat_index


## Actual model stuff
# Table
class Entry :
    def __init__(self):
        self.type = -1
        self.zeros = -1
        self.parent_index = -1
        self.offset = -1


entries = []
for i in range(table_offset, table_offset+table_count*0x14, 0x14):
    e = Entry()
    e.type = gf.get_uint32(fb, i)
    e.zeros = gf.get_uint32(fb, i+4)
    e.parent_index = gf.get_int32(fb, i+8)
    e.offset = gf.get_uint32(fb, i+0x10)
    if i == table_offset:
        data_offset -= e.offset
    e.offset += data_offset
    entries.append(e)


class Block:
    def __init__(self):
        self.offset = -1
        self.size = -1
        self.type = -1


index_blocks = []
vertex_blocks = []
cur_off = entries[0].offset
for e in entries:
    b = Block()
    if e.type == 1:
        b.unk0x00 = gf.get_uint32(fb, cur_off)
        b.unk0x04 = gf.get_uint32(fb, cur_off+4)
        b.unk0x08 = gf.get_uint32(fb, cur_off+8)
        b.vertex_type = gf.get_uint32(fb, cur_off+0xC)
        b.unk0x10 = gf.get_uint32(fb, cur_off + 0x10)
        b.vertex_stride = gf.get_uint16(fb, cur_off + 0x14)
        b.unk0x16 = gf.get_uint16(fb, cur_off + 0x16)
        b.vertex_count = gf.get_uint32(fb, cur_off + 0x18)
        b.offset = gf.get_uint32(fb, cur_off + 0x1C)
        b.size = gf.get_uint32(fb, cur_off + 0x20)
        b.type = gf.get_uint32(fb, cur_off + 0x24)
        unk0x28 = gf.get_uint32(fb, cur_off + 0x28)
        vertex_blocks.append(b)
        cur_off += 0x50
    elif e.type == 2:
        b.unk0x00 = gf.get_uint32(fb, cur_off)
        b.unk0x04 = gf.get_uint32(fb, cur_off+4)
        b.unk0x08 = gf.get_uint32(fb, cur_off+8)
        b.unk0x0C = gf.get_uint32(fb, cur_off+0xC)
        b.index_count = gf.get_uint16(fb, cur_off + 0x10)
        b.offset = gf.get_uint32(fb, cur_off + 0x14)
        b.size = gf.get_uint32(fb, cur_off + 0x18)
        b.type = gf.get_uint32(fb, cur_off + 0x1C)
        index_blocks.append(b)
        cur_off += 0x48
    else:
        continue


## Parsing data

# combining all the chunks together into one big one
chunk_data_map = {}
for i, chunk in enumerate([x for x in os.listdir(folder_path) if ".chunk" in x]):  # praying they read in order - TODO check this with a large file >10 or >100 chunks
    cdata = open(f"{folder_path}/{chunk}", "rb").read()
    index = int(chunk[:-1].split(".chunk")[-1])
    chunk_data_map[index] = cdata

chunk_data = b""
for i in range(len(chunk_data_map.keys())):
    chunk_data += chunk_data_map[i]


class Mesh:
    def __init__(self):
        self.vert_pos = []
        self.vert_uv0 = []
        self.vert_uv1 = []
        self.vert_norm = []
        self.vert_tangent = []
        self.faces = []
        self.name = ""
        self.weight_indices = []
        self.weights = []
        self.weight_pairs = []
        self.parts = []


def parse_index_data(m, ib, b_stride4):
    if b_stride4:
        for j in range(ib.offset, ib.offset+ib.size, 12):
            m.faces.append([gf.get_uint32(chunk_data, j+k*4) for k in range(3)])
    else:
        for j in range(ib.offset, ib.offset+ib.size, 6):
            m.faces.append([gf.get_uint16(chunk_data, j+k*2) for k in range(3)])


def interpolate_coords(verts, scale_min_max):
    it = len(verts[0])
    for i in range(len(verts)):
        verts[i] = [verts[i][j] * scale_min_max[j][-1] + scale_min_max[j][0] for j in range(it)]


def convert(x):
    if x >= 512:
        x -= 1024
    return x


def read_udecn4(data):
    """
    We get an A B C data components of 10 bits each and a D for some metadata
    D tells us about which XYZW component was dropped
    This component is then calculated by assuming normalisation and doing sqrt(1-a^2-b^2-c^2) to get other component
    w component stores which value was dropped. w can hold values 0, 1, 2, 3 which indicates which component was dropped
    """
    return [0, 0, 0]
    data = gf.get_uint32(data, 0)
    m = ((data & 0x3FF) / 1023 - 0.5) / 2**0.5
    n = ((((data >> 10) & 0x3FF) - 0.5) / 1023) / 2**0.5
    o = ((((data >> 20) & 0x3FF) - 0.5) / 1023) / 2**0.5
    p = (data >> 30)
    other_comp = (1-m**2-n**2-o**2)**0.5
    if p == 0:
        quat = [other_comp, m, n, o]
    elif p == 1:
        quat = [m, other_comp, n, o]
    elif p == 2:
        quat = [m, n, other_comp, o]
    else:
        quat = [m, n, o, other_comp]
    direction = scipy.spatial.transform.Rotation.from_quat(quat).apply([100, 0, 0]).tolist()
    return direction


def parse_vertex_data(m, vbs):
    for vb in vbs:
        if vb.vertex_type == 0:  # Position
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                vert = [gf.get_float16(chunk_data, j + k * 2, signed=False) for k in range(3)]
                m.vert_pos.append(vert)
            interpolate_coords(m.vert_pos, model_scale)
        elif vb.vertex_type == 1:  # UV0
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                vert = [gf.get_float16(chunk_data, j + k * 2, signed=False) for k in range(2)]
                m.vert_uv0.append([vert[0], -vert[1]+1])
            interpolate_coords(m.vert_uv0, uv0_scale)
            for i in range(len(m.vert_uv0)):
                m.vert_uv0[i][1] -= uv0_scale[1][0]*2 + uv0_scale[1][2] - 1
        elif vb.vertex_type == 2:  # UV1
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                vert = [gf.get_float16(chunk_data, j + k * 2, signed=False) for k in range(2)]
                m.vert_uv1.append(vert)
            interpolate_coords(m.vert_uv1, uv1_scale)
            for i in range(len(m.vert_uv1)):
                m.vert_uv1[i][1] -= uv1_scale[1][0]*2 + uv1_scale[1][2] - 1  # this doesnt actually work
        elif vb.vertex_type == 5:
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                vert = read_udecn4(chunk_data[j:j+4])
                m.vert_norm.append(vert)
        elif vb.vertex_type == 6:
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                #vert = read_udecn4(chunk_data[j:j+4])
                vert = [0, 0, 0]
                m.vert_tangent.append(vert)
        elif vb.vertex_type == 7:
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                weight_indices = [x for x in chunk_data[j:j+4]]
                m.weight_indices.append(weight_indices)
        elif vb.vertex_type == 8:
            continue
            for j in range(vb.offset, vb.offset + vb.size, vb.vertex_stride):
                weights = [x for x in chunk_data[j:j+4]]
                # s = sum(weights)
                s = 128
                if s != 0:
                    norm_weights = [x/s for x in weights]
                else:
                    norm_weights = [0, 0, 0, 0]
                m.weights.append(norm_weights)

    # Dealing with weights if they exist
    if m.weight_indices:
        if m.weights:
            for i in range(len(m.weight_indices)):
                ind = []
                wei = []
                for j in range(4):
                    if m.weights[i][j]:
                        ind.append(m.weight_indices[i][j])
                        wei.append(m.weights[i][j])
                m.weight_pairs.append([ind, wei])
            #m.weight_pairs = [[[m.weight_indices[i][j], m.weights[i][j]] for j in range(4) if m.weights[i][j] != 0] for i in range(len(m.weight_indices))]
            a = 0
        else:
            m.weight_pairs = [[[x[0]], [1.]] for x in m.weight_indices]

# Grouping into meshes

# TEST: presume the faces are optimised to use up all the vertices so max(index) = len(vertices)
vertex_lens = [x.size/8 for x in vertex_blocks if x.vertex_type == 0]
# last_index = 0
# for i, x in enumerate(vertex_blocks):
#     if x.vertex_type == 0:
#         print(f"Gap of {i-last_index}")
#         last_index = i
choose = [0]
b_choose = False
meshes = []
# vertex_blocks_per_mesh = int(len(vertex_blocks)/len(index_blocks))
current_vb = 0
for i, index_block in enumerate(index_blocks):
    mesh = Mesh()
    vblocks = [vertex_blocks[current_vb]]
    for j in range(current_vb+1, len(vertex_blocks)):
        if vertex_blocks[j].vertex_type == 0:
            current_vb = j
            break
        else:
            vblocks.append(vertex_blocks[j])
    if i not in choose and b_choose:
        continue
    parse_vertex_data(mesh, vblocks)
    parse_index_data(mesh, index_block, len(mesh.vert_pos)>=0x10000)
    # vnt = [[mesh.vert_pos[i], mesh.vert_norm[i], mesh.vert_tangent[i]] for i in range(len(mesh.vert_pos))]
    a = 0
    b = 984
    # mesh.vert_pos = mesh.vert_pos[a:b]
    # mesh.vert_uv0 = mesh.vert_uv0[a:b]
    # mesh.vert_uv1 = mesh.vert_uv1[a:b]
    # mesh.vert_norm = mesh.vert_norm[a:b]
    # mesh.vert_tangent = mesh.vert_tangent[a:b]
    meshes.append(mesh)
    # Save specific index vertex buffers
    # if True and i == 0:
    #     os.makedirs(f"Z:\RE_OtherGames\HI\dump_buffers/{i}/", exist_ok=True)
    #     for j, vb in enumerate(vblocks):
    #         with open(f"Z:\RE_OtherGames\HI\dump_buffers/{i}/{i}_{j}_{vb.vertex_stride}_{vb.vertex_type}.bin", "wb") as f:
    #             f.write(chunk_data[vb.offset:vb.offset+vb.size])


# Writing out model data
# os.makedirs(f"Z:/RE_OtherGames/HI/models/{item_name}/", exist_ok=True)
# lod stuff
last_face_count = -1
last_vert_count = -1
b_lod = True

def trim_verts_data(verts, dsort):
    v_new = []
    for i in dsort:
        v_new.append(verts[i])
    return v_new

# matching meshes wih their parts
for mesh in meshes:
    for part in parts:
        if part[-1].index_offset + part[-1].index_count == len(mesh.faces)*3:
            mesh.parts = part
            parts.remove(part)
            break

for i, mesh in enumerate(meshes):
    if not b_choose and b_lod:
        if len(mesh.faces) < last_face_count and len(mesh.vert_pos) < last_vert_count:
            last_face_count = len(mesh.faces)
            last_vert_count = len(mesh.vert_pos)
            continue
    last_face_count = len(mesh.faces)
    last_vert_count = len(mesh.vert_pos)
    mesh.name = f"{item_name}_{i}_{len(mesh.faces)}_{len(mesh.vert_pos)}"

    # Splitting into parts based on materials
    if mesh.parts:
        for p, part in enumerate(mesh.parts):
            m = Mesh()
            m.faces = copy.deepcopy(mesh.faces[int(part.index_offset / 3):int((part.index_offset + part.index_count) / 3)])

            dsort = set()
            for face in m.faces:
                for f in face:
                    dsort.add(f)
            dsort = sorted(dsort)

            # Moves all the faces down to start at 0
            d = dict(zip(dsort, range(max(dsort) + 1)))
            for j in range(len(m.faces)):
                for k in range(3):
                    m.faces[j][k] = d[m.faces[j][k]]

            m.vert_pos = trim_verts_data(mesh.vert_pos, dsort)
            m.vert_uv0 = trim_verts_data(mesh.vert_uv0, dsort)
            # if mesh.weight_pairs:
            #     m.weight_pairs = trim_verts_data(mesh.weight_pairs, dsort)
            if mesh.vert_uv1:
                m.vert_uv1 = trim_verts_data(mesh.vert_uv1, dsort)
            if mesh.vert_norm:
                m.vert_norm = trim_verts_data(mesh.vert_norm, dsort)
            if part.mat_string:
                m.name = part.mat_string.split('/')[-1] + f"_{p}"
            else:
                m.name = mesh.name + f"_{p}"
            model.add(m)
    else:
        model.add(mesh)


# model.export(f"Z:/RE_OtherGames/HI/models/{item_name}/{item_name}.fbx")
model.export(f"Z:/RE_OtherGames/HI/models/{item_name}.fbx")

a = 0

# with open(f"Z:/RE_OtherGames/HI/models/{item_name}/{item_name}.txt", "w") as f:
#     fo