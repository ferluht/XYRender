import bpy
from math import *
import bpy_extras
import mathutils
import numpy as np
from bpy.app.handlers import persistent
import os.path

import pip
def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

try:
    import scipy
    from scipy.io import wavfile
except:
    install('scipy')
finally:
    import scipy
    from scipy.io import wavfile

def project_3d_to_2d(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    scale = 1.0
    return mathutils.Vector((co_2d.x * scale - scale / 2, co_2d.y * scale - scale / 2))


def on_frame_handler(frame_num, outfile, rewrite=True, sample_rate=96000, fps=30.0):
    lines = []
    samples_per_frame = int(sample_rate / fps)
    sce = bpy.context.scene
    sce.frame_set(frame_num)
    cam = bpy.context.scene.camera
    for obj in bpy.data.objects:
        if (obj.type == 'MESH'):
            me = obj.data

            # Build lookup dictionary for edge keys to edges
            edges = me.edges
            
            for edge in edges:
                p1 = np.array(project_3d_to_2d(cam, obj.matrix_world @ me.vertices[edge.vertices[0]].co))
                p2 = np.array(project_3d_to_2d(cam, obj.matrix_world @ me.vertices[edge.vertices[1]].co))
                lines.append([p1, p2])
        
        if (obj.type == 'GPENCIL'):
            gp = obj.data
            
            for layer in gp.layers:
                frame = layer.active_frame
                if frame is None:
                    continue
                for stroke in frame.strokes:
                    line = []
                    for point in stroke.points:
                        line.append(np.array(project_3d_to_2d(cam, obj.matrix_world @ point.co)))
                    lines.append(line)
    
    l = 0         
    for line in lines:
        for p1, p2 in zip(line[:-1], line[1:]):
            l += np.linalg.norm(p2-p1)
    step = l / samples_per_frame
    frame_data = np.zeros((int(samples_per_frame*2),2), dtype=np.float32)
        
    residual = 0
    pos = 0
    stop = False
    for line in lines:
        for p1, p2 in zip(line[:-1], line[1:]):
            l0 = np.linalg.norm(p2-p1)
            i = 0
            while (step * i - residual <= l0):
                frame_data[pos,:] = p1 + (step * i - residual) * (p2 - p1) / l0
                i += 1
                pos += 1
            residual = l0 - (step * (i - 1) - residual)
    
    frame_data = frame_data[:pos]

    if (rewrite == False) and os.path.isfile(outfile):
        samplerate, audiodata = wavfile.read(outfile)
        frame_data = np.concatenate([audiodata, frame_data])
    wavfile.write(outfile, sample_rate, frame_data)

class XYRenderPropertyGroup(bpy.types.PropertyGroup):
    frequency: bpy.props.FloatProperty(name='frequency', soft_min=0, soft_max=10000)
    start_frame: bpy.props.IntProperty(name='start frame', soft_min=0, soft_max=100)
    end_frame: bpy.props.IntProperty(name='end frame', soft_min=0, soft_max=100)
    rewrite_file: bpy.props.BoolProperty(name='rewrite')
    outfile: bpy.props.StringProperty(name='output file')

class XYRenderToolShelf(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = 'XYRender'
    bl_context = 'objectmode'
    bl_category = 'XYRender'
    
    def draw(self, context):
        layout = self.layout
        subrow = layout.row(align=True)
        subrow.prop(context.scene.custom_props, 'start_frame')
        subrow.prop(context.scene.custom_props, 'end_frame')
        subrow = layout.row(align=True)
        subrow.prop(context.scene.custom_props, 'outfile')
        subrow.prop(context.scene.custom_props, 'rewrite_file')
        layout.operator('xyrender.render', text = 'Render!')
        

class XYRender(bpy.types.Operator):
    bl_idname = 'xyrender.render'
    bl_label = 'XYRenderOperator'
    bl_options = {'INTERNAL'}

    @classmethod
    def poll(cls, context):
        return context.object is not None
    
    def execute(self, context):
        props = self.properties
        scene = context.scene
        for f in range(scene.custom_props.start_frame, scene.custom_props.end_frame):
            on_frame_handler(f, scene.custom_props.outfile, 
                            rewrite=bool(f==scene.custom_props.start_frame and scene.custom_props.rewrite_file))
            self.report({'INFO'}, f"Rendered frame {f}")
        return {'FINISHED'}

bl_info={
        "name": "XYRender",
        "category": "Rendering",
        "author": "ferluht @ ed9m",
        "blender": (2,93,0),
        "location": "View3D > right-side panel > XYRender",
        "description":"Addon to convert 2d or 3d animation to XY format and render it to a stereo audio file",
        "warning": "",
        "wiki_url":"https://github.com/ed9m/XYRender",
        "tracker_url": "",
        "version":(0,1,1)
    }

def register():
    bpy.utils.register_class(XYRenderPropertyGroup)
    bpy.types.Scene.custom_props = bpy.props.PointerProperty(type=XYRenderPropertyGroup)
    bpy.utils.register_class(XYRender)
    bpy.utils.register_class(XYRenderToolShelf)


def unregister():
    del bpy.types.Scene.custom_props 
    bpy.utils.unregister_class(XYRenderPropertyGroup)
    bpy.utils.unregister_class(XYRender)
    bpy.utils.unregister_class(XYRenderToolShelf)     

if __name__ == '__main__':
    register()