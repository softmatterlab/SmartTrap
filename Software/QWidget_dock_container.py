from PyQt6.QtWidgets import QDockWidget

class QWidgetWindowDocker(QDockWidget):
    def __init__(self, Qwidget, Title="Widget container"):
        super().__init__(Title)
        self.widget = Qwidget
        self.setWidget(self.widget)  # Embed the original widget inside the QDockWidget


    def closeEvent(self, event):
        # Instead of closing, hide the dock widget
        event.ignore()
        self.hide()