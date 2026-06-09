import qt


def createButton(name, callback=None, isCheckable=False, icon=None, toolTip="", parent=None):
    """Create a QPushButton with text, optional click callback, checkable state, icon and tooltip."""
    button = qt.QPushButton(name, parent)
    if callback is not None:
        button.connect("clicked(bool)", callback)
    if icon:
        button.setIcon(icon)
    button.setCheckable(isCheckable)
    button.setToolTip(toolTip)
    return button


def addInCollapsibleLayout(childWidget, parentLayout, collapsibleText, isCollapsed=True):
    """Wrap childWidget in a collapsible button attached to parentLayout."""
    import ctk

    collapsibleButton = ctk.ctkCollapsibleButton()
    collapsibleButton.text = collapsibleText
    collapsibleButton.collapsed = isCollapsed
    parentLayout.addWidget(collapsibleButton)
    layout = qt.QVBoxLayout()
    layout.addWidget(childWidget)
    collapsibleButton.setLayout(layout)
