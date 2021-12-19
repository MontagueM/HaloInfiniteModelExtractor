import fbx
import sys
import os
class Model:
    def __init__(self):
        self.manager = fbx.FbxManager.Create()
        if not self.manager:
            sys.exit(0)

        self.ios = fbx.FbxIOSettings.Create(self.manager, fbx.IOSROOT)
        self.exporter = fbx.FbxExporter.Create(self.manager, '')
        self.scene = fbx.FbxScene.Create(self.manager, '')
        self.face_counter = 0
        self.bones = []

    def add(self, submesh,  direc="", b_unreal=False):
        node, mesh = self.create_mesh(submesh)

        if not mesh.GetLayer(0):
            mesh.CreateLayer()
        if submesh.vert_uv1:
            mesh.CreateLayer()
        layer = mesh.GetLayer(0)

        # if submesh.material:
        #     if submesh.diffuse:
        #         self.apply_diffuse(submesh.diffuse, f'{direc}/textures/{submesh.diffuse}.dds', node)
        #         node.SetShadingMode(fbx.FbxNode.eTextureShading)

        if submesh.vert_uv0 or submesh.vert_uv1:
            if submesh.vert_uv0:
                self.create_uv(mesh, submesh, layer, "uv0")
            if submesh.vert_uv1:

                self.create_uv(mesh, submesh, mesh.GetLayer(1), "uv1")
        if submesh.vert_norm:
            self.add_norm(mesh, submesh, layer)
        # if submesh.vert_col:
        #     self.add_vert_colours(mesh, submesh, layer)

        if submesh.weight_pairs:
            self.add_weights(mesh, submesh, "")

        # if not b_unreal and not submesh.weight_pairs:
        #     node.LclRotation.Set(fbx.FbxDouble3(-90, 180, 0))

        self.scene.GetRootNode().AddChild(node)

    def export(self, save_path=None, ascii_format=False):
        """Export the scene to an fbx file."""

        os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
        if not self.manager.GetIOSettings():
            self.ios = fbx.FbxIOSettings.Create(self.manager, fbx.IOSROOT)
            self.manager.SetIOSettings(self.ios)

        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_MATERIAL, True)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_TEXTURE, True)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_EMBEDDED, False)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_SHAPE, True)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GOBO, False)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_ANIMATION, True)
        self.manager.GetIOSettings().SetBoolProp(fbx.EXP_FBX_GLOBAL_SETTINGS, True)
        if ascii_format:
            b_ascii = 1
        else:
            b_ascii = -1
        self.exporter.Initialize(save_path, b_ascii, self.manager.GetIOSettings())
        self.exporter.Export(self.scene)
        self.exporter.Destroy()

    def create_mesh(self, submesh):
        mesh = fbx.FbxMesh.Create(self.scene, submesh.name)
        controlpoints = [fbx.FbxVector4(-x[0]*100, x[2]*100, x[1]*100) for x in submesh.vert_pos]
        for i, p in enumerate(controlpoints):
            mesh.SetControlPointAt(p, i)
        for face in submesh.faces:
            mesh.BeginPolygon()
            mesh.AddPolygon(face[0])
            mesh.AddPolygon(face[1])
            mesh.AddPolygon(face[2])
            mesh.EndPolygon()
        node = fbx.FbxNode.Create(self.scene, submesh.name)
        node.SetNodeAttribute(mesh)
        return node, mesh

    def add_norm(self, mesh, submesh, layer):
        # Dunno where to put this, norm quat -> norm vec conversion
        return
        normElement = fbx.FbxLayerElementNormal.Create(mesh, 'norm')
        normElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
        normElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
        for i, vec in enumerate(submesh.vert_norm):
            normElement.GetDirectArray().Add(fbx.FbxVector4(-vec[0], vec[2], vec[1]))
        layer.SetNormals(normElement)

    def create_uv(self, mesh, submesh, layer, uv_name):
        uvDiffuseLayerElement = fbx.FbxLayerElementUV.Create(mesh, '{} {}'.format(uv_name, submesh.name))
        uvDiffuseLayerElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
        uvDiffuseLayerElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
        if uv_name == "uv0":
            for i, p in enumerate(submesh.vert_uv0):
                uvDiffuseLayerElement.GetDirectArray().Add(fbx.FbxVector2(p[0], p[1]))
        elif uv_name == "uv1":
            for i, p in enumerate(submesh.vert_uv1):
                uvDiffuseLayerElement.GetDirectArray().Add(fbx.FbxVector2(p[0], p[1]))
        layer.SetUVs(uvDiffuseLayerElement, fbx.FbxLayerElement.eTextureDiffuse)

    def add_vert_colours(self, mesh, submesh, layer):
        vertColourElement = fbx.FbxLayerElementVertexColor.Create(mesh, 'colour')
        vertColourElement.SetMappingMode(fbx.FbxLayerElement.eByControlPoint)
        vertColourElement.SetReferenceMode(fbx.FbxLayerElement.eDirect)
        for i, p in enumerate(submesh.vertex_colour):
            vertColourElement.GetDirectArray().Add(fbx.FbxColor(p[0], p[1], p[2], p[3]))
        layer.SetVertexColors(vertColourElement)

    def apply_diffuse(self, tex_name, tex_path, node):
        """Bad function that shouldn't be used as shaders should be, but meh"""
        lMaterialName = 'mat {}'.format(tex_name)
        lMaterial = fbx.FbxSurfacePhong.Create(self.scene, lMaterialName)
        lMaterial.DiffuseFactor.Set(1)
        lMaterial.ShadingModel.Set('Phong')
        node.AddMaterial(lMaterial)

        gTexture = fbx.FbxFileTexture.Create(self.scene, 'Diffuse Texture {}'.format(tex_name))
        lTexPath = tex_path
        gTexture.SetFileName(lTexPath)
        gTexture.SetRelativeFileName(lTexPath)
        gTexture.SetTextureUse(fbx.FbxFileTexture.eStandard)
        gTexture.SetMappingType(fbx.FbxFileTexture.eUV)
        gTexture.SetMaterialUse(fbx.FbxFileTexture.eModelMaterial)
        gTexture.SetSwapUV(False)
        gTexture.SetTranslation(0.0, 0.0)
        gTexture.SetScale(1.0, 1.0)
        gTexture.SetRotation(0.0, 0.0)

        if lMaterial:
            lMaterial.Diffuse.ConnectSrcObject(gTexture)
        else:
            raise RuntimeError('Material broken somewhere')

    def add_temp_bones(self):
        for i in range(100):
            nodeatt = fbx.FbxSkeleton.Create(self.scene, str(i))
            nodeatt.SetSkeletonType(fbx.FbxSkeleton.eLimbNode)
            fbxnode = fbx.FbxNode.Create(self.scene, str(i))
            fbxnode.SetNodeAttribute(nodeatt)
            # fbxnode.LclRotation.Set(fbx.FbxDouble3(-90, 0, 180))
            self.bones.append(fbxnode)
            self.scene.GetRootNode().AddChild(fbxnode)

    def add_weights(self, mesh, submesh, name):
        self.add_temp_bones()
        skin = fbx.FbxSkin.Create(self.scene, name)
        bone_cluster = []
        for bone in self.bones:
            def_cluster = fbx.FbxCluster.Create(self.scene, 'BoneWeightCluster')
            def_cluster.SetLink(bone)
            def_cluster.SetLinkMode(fbx.FbxCluster.eTotalOne)
            bone_cluster.append(def_cluster)

            transform = bone.EvaluateGlobalTransform()
            def_cluster.SetTransformLinkMatrix(transform)

        for i, w in enumerate(submesh.weight_pairs):
            indices = w[0]
            weights = w[1]
            for j in range(len(indices)):
                if len(bone_cluster) < indices[j]:
                    print(
                        'Bone index longer than bone clusters, could not add weights ({} > {})'.format(indices[j],len(bone_cluster) ))
                    return
                try:
                    bone_cluster[indices[j]].AddControlPointIndex(i, weights[j])
                except IndexError:
                    pass

        for c in bone_cluster:
            skin.AddCluster(c)

        mesh.AddDeformer(skin)