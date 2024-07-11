# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:28:23 2023

@author: marti
"""
import abc

class MouseInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        # TODO Add a icon or similar bar
        return (hasattr(subclass, 'mouseMove') and
                callable(subclass.mouseMove) and
                hasattr(subclass, 'mousePress') and
                callable(subclass.mousePress) and
                hasattr(subclass, 'mouseRelease') and
                callable(subclass.mouseRelease) and
                hasattr(subclass, 'draw') and
                callable(subclass.draw) and
                hasattr(subclass, 'getToolName') and
                callable(subclass.getToolName) and
                hasattr(subclass, 'getToolTip') and
                callable(subclass.getToolTip) and
                hasattr(subclass, 'mouseDoubleClick') and
                callable(subclass.mouseDoubleClick) or
                NotImplemented)
                
    

class MarkPositionTool(MouseInterface):
    
    def __init__(self, c_p):
        self.c_p = c_p
        self.marker_positions = [] # Marker positions in image coordinates
        self.distance_limit = 30
        # TODO send in central position
    def resize_coordinate(self):
        pass
    def mousePressLeft(self):
        # Adds a marker in specified position
        x_pos = self.c_p['mouse_params'][1] * self.c_p['image_scale']
        y_pos = self.c_p['mouse_params'][2] * self.c_p['image_scale']
        self.marker_positions.append(x_pos, y_pos)
        
    def mousePressRight(self):
        # Remove a marker
        if len(self.marker_positions) < 1:
            return
        min_distance = 80000
        min_x = -1
        min_y = -1
        # Find closest item
        click_x = self.c_p['mouse_params'][1] * self.c_p['image_scale']
        click_y = self.c_p['mouse_params'][2] * self.c_p['image_scale']
        for x,y in zip(self.marker_positions):
            distance = (x - click_x)**2 + (y - click_y) ** 2
            if distance < min_distance:
                min_distance = distance
                min_x = click_x
                min_y = click_y
        # Pop item
    
    def draw(self, qp):
        if len(self.marker_positions) < 1:
            return
        for x,y in zip(self.marker_positions):
            rx=50
            ry=50
            self.qp.drawEllipse(x, y, rx, ry)

