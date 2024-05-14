import os
import pybullet as p
import numpy as np


class Object(object):
    def __init__(self):
        self.body_id = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return self.body_id
        self.body_id = self._load()
        self.loaded = True
        self.n_joints = p.getNumJoints(self.body_id)
        return self.body_id

    def get_position_orientation(self):
        if self.n_joints: # interactive objects
            info = []
            for link in range(self.n_joints):
                data = p.getLinkState(self.body_id, link)
                info.append(list(data[0]) + list(data[1]))
        else:
            pos, orien = p.getBasePositionAndOrientation(self.body_id)
            info = list(pos) + list(orien)
        return info

    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)


class YCBObject(Object):
    def __init__(self, name, scale=1):
        super(YCBObject, self).__init__()
        self.visual_filename = os.path.join('assets', 'ycb', name,
                                            'textured_simple.obj')
        self.collision_filename = os.path.join('assets', 'ycb', name,
                                               'textured_simple_vhacd.obj')
        self.scale = scale

    def _load(self):
        collision_id = p.createCollisionShape(p.GEOM_MESH,
                                              fileName=self.collision_filename,
                                              meshScale=self.scale)
        visual_id = p.createVisualShape(p.GEOM_MESH,
                                        fileName=self.visual_filename,
                                        meshScale=self.scale)

        body_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                    baseVisualShapeIndex=visual_id,
                                    basePosition=[0.2, 0.2, 0.1],
                                    baseMass=0.1)
        return body_id


class InteractiveObj(Object):
    def __init__(self, filename, scale=1):
        super(InteractiveObj, self).__init__()
        self.filename = filename
        self.scale = scale

    def _load(self):
        body_id = p.loadURDF(self.filename, globalScaling=self.scale,
                             flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=True)
        self.mass = p.getDynamicsInfo(body_id, -1)[0]

        return body_id


class RBOObject(InteractiveObj):
    def __init__(self, name, scale=1):
        filename = os.path.join('assets', 'rbo', name, 'configuration',
                                '{}.urdf'.format(name))
        super(RBOObject, self).__init__(filename, scale)
