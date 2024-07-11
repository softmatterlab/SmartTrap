from CustomMouseTools import MouseInterface
from PyQt6.QtGui import QPen, QColor, QBrush, QCursor

class CameraMeasurements(MouseInterface):
    """
    Allows the user to measure distances on the camera.
    Left click for first position and right click for second position.
    Positions are marked with small circles.
    """
    def __init__(self, c_p ):
        self.c_p = c_p
        self.x_prev_A = 0
        self.y_prev_A = 0
        self.x_prev_B = 0
        self.y_prev_B = 0
        self.red_pen = QPen(QColor(255,0,0))
        self.blue_pen = QPen(QColor(0,0,255))
        self.circle_radii = 6

    def mousePress(self):
        # left click
        if self.c_p['mouse_params'][0] == 1:
            self.x_prev_A = self.c_p['mouse_params'][1]
            self.y_prev_A = self.c_p['mouse_params'][2]
        # Right click
        if self.c_p['mouse_params'][0] == 2:
            self.x_prev_B = self.c_p['mouse_params'][1]
            self.y_prev_B = self.c_p['mouse_params'][2]
        dx = self.c_p['image_scale']*(self.x_prev_B - self.x_prev_A) * self.c_p['microns_per_pix']
        dy = self.c_p['image_scale']*(self.y_prev_B - self.y_prev_A) * self.c_p['microns_per_pix']

        print(f"CLick - dx: {dx:.3f}, dy: {dy:.3f}, length {((dx**2 + dy**2)**0.5):.3f} [microns]")        
        
    def mouseRelease(self):
        if self.c_p['mouse_params'][0] == 2:
            pass
        
    def mouseDoubleClick(self):
        pass
    
    def draw(self, qp):
        # TODO add draw here to show exactly where the click was
        qp.setPen(self.red_pen)

        qp.drawEllipse(self.x_prev_A-self.circle_radii, self.y_prev_A-self.circle_radii, self.circle_radii*2,self.circle_radii*2)
        
        if self.x_prev_B is not None:
            qp.setPen(self.blue_pen)
            qp.drawEllipse(self.x_prev_B-self.circle_radii, self.y_prev_B-self.circle_radii, self.circle_radii*2,self.circle_radii*2)

    def mouseMove(self):
        dx = self.c_p['image_scale']*(self.c_p['mouse_params'][3] - self.x_prev_A) * self.c_p['microns_per_pix']
        dy = self.c_p['image_scale']*(self.c_p['mouse_params'][4] - self.y_prev_A) * self.c_p['microns_per_pix']
        print(f"dx: {dx:.3f}, dy: {dy:.3f}, length {((dx**2 + dy**2)**0.5):.3f} [microns]")

        if self.c_p['mouse_params'][0] == 1:
            pass

        if self.c_p['mouse_params'][0] == 2:
            pass

    def getToolName(self):
        return "Camera ruler"

    def getToolTip(self):
        return "Measures distances on the camera. Left click for first position and right click for second position.\n Drag and left click to measure continously."
        