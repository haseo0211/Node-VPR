"""Node-VPR Node Map Editor v0.7.

PySide6 desktop editor. No physical coordinates are assumed:
- nodes have stable internal IDs and editable display labels;
- canvas coordinates are schematic;
- metric scale is optional and manually calibrated;
- a measured edge length is distinct from a drawing-derived estimate.
v0.7: E1/E2/... entrance stubs are visual markers, NOT routing graph nodes/edges.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from pathlib import Path

# PyVistaQt uses QtPy. Prefer the same binding as our PySide6 GUI.
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPen, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolBar,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Lazy 3D initialization: QtInteractor itself is created only when 3D is opened.
# PyVista / PyVistaQt are installed in the user's virtual environment.
try:
    import numpy as np
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PV_ERROR = None
except Exception as exc:
    np = None
    pv = None
    QtInteractor = None
    PV_ERROR = str(exc)


GRID = 25  # Canvas grid step. Display and snap are separate options.
SMART_GUIDE_TOLERANCE_PX = 9  # Drag alignment tolerance, measured on screen.
ENTRANCE_LENGTH = 65.0  # Schematic only, not a surveyed length.
ENTRANCE_COLOR = "#dc2626"


class MapHistoryCommand(QUndoCommand):
    """A document snapshot stored only when a user edit is completed.

    QUndoStack calls redo() as soon as a command is pushed. Because the edit
    already happened, that first call must *not* rebuild the scene; later
    redo calls restore the recorded post-edit state.
    """

    def __init__(self, editor: "MapEditor", description: str, before: dict, after: dict):
        super().__init__(description)
        self.editor = editor
        self.before = copy.deepcopy(before)
        self.after = copy.deepcopy(after)
        self.first_redo = True

    def undo(self):
        self.editor.restore_snapshot(self.before)

    def redo(self):
        if self.first_redo:
            self.first_redo = False
            return
        self.editor.restore_snapshot(self.after)


class NodeItem(QGraphicsEllipseItem):
    """A movable node; its internal ID does not change when the label changes."""

    def __init__(self, editor: "MapEditor", node: dict, canvas: "Canvas"):
        super().__init__(-19, -19, 38, 38)
        self.editor = editor
        self.canvas = canvas
        self.node_id = node["id"]  # Internal stable ID, e.g. U0001
        self.text = QGraphicsSimpleTextItem("", self)
        self.text.setBrush(QBrush(QColor("white")))
        self.text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setPen(QPen(QColor("#1d4ed8"), 2))
        self.setZValue(2)
        self.refresh_label()

        # Set the initial position BEFORE enabling position-change notifications.
        self.setPos(QPointF(float(node["x"]), float(node["y"])))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

    def refresh_label(self):
        node = self.editor.nodes[self.node_id]
        label = node.get("node_id", "")
        self.setBrush(QBrush(QColor("#2563eb" if label else "#d97706")))
        self.text.setText(label or "?")
        bounds = self.text.boundingRect()
        self.text.setPos(-bounds.width() / 2, -bounds.height() / 2)
        cross_floor = []
        for edge in self.editor.edges:
            other_id = None
            if edge["from"] == self.node_id:
                other_id = edge["to"]
            elif edge["to"] == self.node_id:
                other_id = edge["from"]
            if other_id is not None and other_id in self.editor.nodes:
                other = self.editor.nodes[other_id]
                if other["floor"] != node["floor"]:
                    cross_floor.append(
                        f"{edge['type']} → {self.editor.node_name(other_id)} ({other['floor']})"
                    )
        # A gold outline marks vertical links which are drawn in the 3D view.
        self.setPen(QPen(QColor("#c98325" if cross_floor else "#1d4ed8"),
                         3 if cross_floor else 2))
        tooltip = f"노드: {label or '(미지정)'}\n내부 ID: {self.node_id}\n더블클릭: 이름 입력"
        if cross_floor:
            tooltip += "\n층간 연결:\n" + "\n".join(cross_floor)
        self.setToolTip(tooltip)

    def mouseDoubleClickEvent(self, event):
        if self.editor.mode == "select":
            self.editor.edit_node_label(self.node_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if (self.editor.mode == "select"
                and event.button() == Qt.MouseButton.LeftButton):
            self.editor.active_canvas = self.canvas
            self.editor.dragging_node_id = self.node_id
            self.editor.begin_node_drag()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.editor.dragging_node_id == self.node_id:
            self.editor.dragging_node_id = None
            self.editor.clear_alignment_guides()
            self.editor.finish_node_drag()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Grid snap has priority when enabled. Smart guides are used only
            # during dragging with grid snap OFF, so grid-aligned nodes always
            # remain on grid intersections.
            if self.editor.snap_to_grid_enabled:
                return self.editor.snap_point_to_grid(value)
            if (self.editor.dragging_node_id == self.node_id
                    and self.editor.smart_guides_enabled):
                return self.editor.align_to_other_nodes(self.node_id, value, self.canvas)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            node = self.editor.nodes.get(self.node_id)
            if node is not None:
                pos = self.pos()
                node["x"], node["y"] = round(pos.x(), 2), round(pos.y(), 2)
                self.editor.update_adjacent_edges(self.node_id, self.canvas)
        return super().itemChange(change, value)


class EdgeItem(QGraphicsLineItem):
    """A line on the map, with separately stored surveyed length if available."""

    def __init__(self, editor: "MapEditor", edge: dict, src: NodeItem, dst: NodeItem):
        super().__init__()
        self.editor = editor
        self.edge_id = edge["id"]
        self.source, self.target = src, dst
        self.setPen(QPen(QColor("#475569"), 3))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(0)
        self.length_text = QGraphicsSimpleTextItem("", self)
        self.length_text.setBrush(QBrush(QColor("#334155")))
        self.length_text.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.update_line()

    def update_line(self):
        self.setLine(QLineF(self.source.pos(), self.target.pos()))
        edge = self.editor.find_edge(self.edge_id)
        if edge is None:
            return
        measured = edge.get("length_m")
        drawn = self.line().length() / GRID
        scale = self.editor.meters_per_grid
        if measured is not None:
            display = f"{float(measured):g} m"
            self.setToolTip("실측 입력 길이: " + display + "\n더블클릭하여 수정")
        elif scale > 0:
            display = f"≈{drawn * scale:.1f} m"
            self.setToolTip("도식상 길이 추정: " + display + " (실측 아님)\n더블클릭하여 실측값 입력")
        else:
            display = ""
            self.setToolTip("길이 미설정\n더블클릭하여 실측값 입력")
        self.length_text.setText(display)
        self.length_text.setVisible(bool(display))
        midpoint = QPointF(
            (self.source.pos().x() + self.target.pos().x()) / 2,
            (self.source.pos().y() + self.target.pos().y()) / 2,
        )
        bounds = self.length_text.boundingRect()
        self.length_text.setPos(midpoint.x() - bounds.width() / 2, midpoint.y() - 20)

    def mouseDoubleClickEvent(self, event):
        if self.editor.mode == "select":
            self.editor.edit_edge_length(self.edge_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class EntranceItem(QGraphicsLineItem):
    """A red external approach marker attached to one real NodeItem.

    Entrance IDs E1/E2/... are a separate namespace from regular edge IDs.
    The two collections are independent: entrances never affect route search.
    """

    def __init__(self, editor: "MapEditor", entrance: dict, source: NodeItem):
        super().__init__()
        self.editor = editor
        self.entrance_id = entrance["id"]
        self.source = source
        self.setPen(QPen(QColor(ENTRANCE_COLOR), 5, Qt.PenStyle.SolidLine,
                         Qt.PenCapStyle.RoundCap))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setZValue(1)
        self.tip = QGraphicsEllipseItem(-5, -5, 10, 10, self)
        self.tip.setPen(QPen(QColor(ENTRANCE_COLOR), 2))
        self.tip.setBrush(QBrush(QColor("#ffffff")))
        self.tip.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.label = QGraphicsSimpleTextItem("", self)
        self.label.setBrush(QBrush(QColor(ENTRANCE_COLOR)))
        self.label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.update_line()

    def update_line(self):
        entry = self.editor.find_entrance(self.entrance_id)
        if entry is None:
            return
        dx, dy = float(entry["dx"]), float(entry["dy"])
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return
        src = self.source.pos()
        tip = QPointF(src.x() + ENTRANCE_LENGTH * dx / norm,
                      src.y() + ENTRANCE_LENGTH * dy / norm)
        self.setLine(QLineF(src, tip))
        self.tip.setPos(tip)
        self.label.setText(self.entrance_id)
        self.label.setPos(tip.x() + 8, tip.y() - 12)
        angle = math.degrees(math.atan2(dy, dx)) % 360
        self.setToolTip(
            f"외부 진입점: {self.entrance_id}\n연결 노드: "
            f"{self.editor.node_name(entry['node'])}\n"
            f"방향: {angle:.1f}° (오른쪽 0°, 아래 90°)\n"
            "더블클릭: 번호/방향 수정 · 선택 후 Delete: 삭제"
        )

    def mouseDoubleClickEvent(self, event):
        if self.editor.mode == "select":
            self.editor.edit_entrance(self.entrance_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class Canvas(QGraphicsView):
    def __init__(self, editor: "MapEditor", scene: QGraphicsScene, floor_getter):
        super().__init__(scene)
        self.editor = editor
        self.floor_getter = floor_getter
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: dict[str, EdgeItem] = {}
        self.entrance_items: dict[str, EntranceItem] = {}
        self.guide_x: float | None = None
        self.guide_y: float | None = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def current_floor(self):
        return self.floor_getter()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#f8fafc"))
        if not self.editor.grid_visible:
            return
        x0 = math.floor(rect.left() / GRID)
        x1 = math.ceil(rect.right() / GRID)
        y0 = math.floor(rect.top() / GRID)
        y1 = math.ceil(rect.bottom() / GRID)
        minor = QPen(QColor("#e3e8ef"), 0)
        major = QPen(QColor("#c2ccd8"), 0)
        for i in range(x0, x1 + 1):
            painter.setPen(major if i % 5 == 0 else minor)
            x = i * GRID
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
        for i in range(y0, y1 + 1):
            painter.setPen(major if i % 5 == 0 else minor)
            y = i * GRID
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))

    def drawForeground(self, painter, rect):
        """PowerPoint-like alignment guides, visible only while dragging."""
        guide_pen = QPen(QColor("#e11d48"), 0, Qt.PenStyle.DashLine)
        painter.setPen(guide_pen)
        if self.guide_x is not None:
            x = self.guide_x
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
        if self.guide_y is not None:
            y = self.guide_y
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))

    def mousePressEvent(self, event):
        self.editor.active_canvas = self
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self.editor.mode == "add":
                # Do not add another node on top of an existing node.
                item = self.itemAt(pos)
                while item is not None and not isinstance(item, NodeItem):
                    item = item.parentItem()
                if item is None:
                    self.editor.create_node(self.mapToScene(pos), self.current_floor())
                event.accept()
                return
            if self.editor.mode == "connect":
                item = self.itemAt(pos)
                while item is not None and not isinstance(item, NodeItem):
                    item = item.parentItem()
                if isinstance(item, NodeItem):
                    self.editor.select_endpoint(item.node_id)
                event.accept()
                return
            if self.editor.mode == "entrance":
                item = self.itemAt(pos)
                while item is not None and not isinstance(item, NodeItem):
                    item = item.parentItem()
                if self.editor.first_endpoint is None:
                    if isinstance(item, NodeItem):
                        self.editor.first_endpoint = item.node_id
                        self.editor.statusBar().showMessage(
                            f"{self.editor.node_name(item.node_id)} 선택: "
                            "출입구가 있는 방향의 빈 공간을 클릭하세요."
                        )
                    else:
                        self.editor.statusBar().showMessage("먼저 연결할 노드를 클릭하세요.")
                elif item is None:
                    self.editor.create_entrance(self.editor.first_endpoint,
                                                self.mapToScene(pos))
                    self.editor.first_endpoint = None
                else:
                    self.editor.statusBar().showMessage(
                        "노드 바깥의 빈 공간을 눌러 출입구 방향을 지정하세요."
                    )
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        self.scale(1.15 if event.angleDelta().y() > 0 else 1 / 1.15,
                   1.15 if event.angleDelta().y() > 0 else 1 / 1.15)
        event.accept()


class FloorPane(QFrame):
    """A second/third/fourth 2D view of the SAME document, one floor per pane."""

    def __init__(self, editor: "MapEditor", index: int):
        super().__init__()
        self.editor = editor
        self.index = index
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { border: 1px solid #d7dde6; } QGraphicsView { border: 0; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(4)
        header = QHBoxLayout()
        self.label = QLabel(f"화면 {index + 1}")
        header.addWidget(self.label)
        header.addStretch(1)
        header.addWidget(QLabel("층:"))
        self.floor_combo = QComboBox()
        self.floor_combo.setMinimumWidth(85)
        header.addWidget(self.floor_combo)
        layout.addLayout(header)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_scene.setSceneRect(-1500, -1000, 3000, 2000)
        self.canvas = Canvas(editor, self.graphics_scene,
                             floor_getter=lambda: self.floor_combo.currentText())
        layout.addWidget(self.canvas, stretch=1)
        self.floor_combo.currentTextChanged.connect(lambda _=None: editor.render_canvas(self.canvas))

    def set_floors(self, floors: list[str], prefer_index: bool = False):
        old = self.floor_combo.currentText()
        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        self.floor_combo.addItems(floors)
        if prefer_index or old not in floors:
            index = min(self.index, len(floors) - 1)
            self.floor_combo.setCurrentIndex(max(0, index))
        else:
            self.floor_combo.setCurrentText(old)
        self.floor_combo.blockSignals(False)
        self.editor.render_canvas(self.canvas)


class MapEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1200, 800)
        self.floors = ["1F"]
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.entrances: list[dict] = []
        self.next_node_uid = 1
        self.next_edge_id = 1
        self.next_entrance_id = 1
        self.current_path: Path | None = None
        self.mode = "select"
        self.first_endpoint: str | None = None
        self.grid_visible = True
        self.smart_guides_enabled = True
        self.snap_to_grid_enabled = False
        self.dragging_node_id: str | None = None
        self.guide_x: float | None = None
        self.guide_y: float | None = None
        self.meters_per_grid = 0.0  # 0 means not calibrated.
        self.drag_start_snapshot: dict | None = None
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(100)  # Keep history bounded for long sessions.

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-1500, -1000, 3000, 2000)
        self.canvas = Canvas(self, self.scene,
                             floor_getter=lambda: self.floor_combo.currentText())
        self.active_canvas = self.canvas
        self.tabs = QTabWidget(self)
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(self.canvas)
        self.tabs.addTab(single_page, "2D · 단일 층")

        self.multi_page = QWidget()
        multi_layout = QVBoxLayout(self.multi_page)
        multi_layout.setContentsMargins(5, 5, 5, 5)
        control = QHBoxLayout()
        control.addWidget(QLabel("동시 보기:"))
        self.pane_count = QComboBox()
        self.pane_count.addItems(["2개", "3개", "4개"])
        self.pane_count.currentIndexChanged.connect(self.update_pane_visibility)
        control.addWidget(self.pane_count)
        control.addSpacing(12)
        control.addWidget(QLabel("엣지 연결 모드에서 서로 다른 층의 노드를 차례로 클릭하면 층간 연결"))
        control.addStretch(1)
        multi_layout.addLayout(control)
        self.pane_grid = QGridLayout()
        self.pane_grid.setSpacing(6)
        self.multi_panes = [FloorPane(self, i) for i in range(4)]
        for i, pane in enumerate(self.multi_panes):
            self.pane_grid.addWidget(pane, i // 2, i % 2)
        self.pane_grid.setRowStretch(0, 1)
        self.pane_grid.setRowStretch(1, 1)
        self.pane_grid.setColumnStretch(0, 1)
        self.pane_grid.setColumnStretch(1, 1)
        multi_layout.addLayout(self.pane_grid, stretch=1)
        self.tabs.addTab(self.multi_page, "2D · 여러 층")

        self.view_3d_page = QWidget()
        self.view_3d_layout = QVBoxLayout(self.view_3d_page)
        self.view_3d_layout.setContentsMargins(4, 4, 4, 4)
        self.info_3d = QLabel(
            "전체 층을 하나의 3D 그래프에 표시합니다. 층별 높이는 시각화용이며 실측 높이가 아닙니다."
        )
        self.view_3d_layout.addWidget(self.info_3d)
        self.viewer_placeholder = QLabel("3D 탭을 처음 열면 PyVista 뷰어를 초기화합니다.")
        self.viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view_3d_layout.addWidget(self.viewer_placeholder, 1)
        self.plotter = None
        self.tabs.addTab(self.view_3d_page, "3D · 전체 맵")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.setCentralWidget(self.tabs)
        self.update_pane_visibility()

        toolbar = QToolBar("편집", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.mode_buttons = {}
        for mode, caption in (("select", "선택 / 이동"), ("add", "노드 추가"),
                              ("connect", "엣지 연결"), ("entrance", "Entrance 추가")):
            button = QPushButton(caption)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, m=mode: self.set_mode(m))
            toolbar.addWidget(button)
            self.mode_buttons[mode] = button
        toolbar.addSeparator()
        button = QPushButton("선택 삭제")
        button.clicked.connect(self.delete_selected)
        toolbar.addWidget(button)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("  층: "))
        self.floor_combo = QComboBox()
        self.floor_combo.addItems(self.floors)
        self.floor_combo.currentTextChanged.connect(self.render_floor)
        toolbar.addWidget(self.floor_combo)
        button = QPushButton("층 추가")
        button.clicked.connect(self.add_floor)
        toolbar.addWidget(button)
        button = QPushButton("층간 연결")
        button.clicked.connect(self.connect_floors)
        toolbar.addWidget(button)
        toolbar.addSeparator()
        button = QPushButton("불러오기")
        button.clicked.connect(self.load_map)
        toolbar.addWidget(button)
        button = QPushButton("저장")
        button.clicked.connect(self.save_map)
        toolbar.addWidget(button)
        toolbar.addSeparator()
        self.undo_button = QPushButton("실행 취소")
        self.undo_button.setToolTip("Ctrl+Z · 직전 편집 되돌리기")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_stack.undo)
        toolbar.addWidget(self.undo_button)
        self.redo_button = QPushButton("다시 실행")
        self.redo_button.setToolTip("Ctrl+Y 또는 Ctrl+Shift+Z")
        self.redo_button.setEnabled(False)
        self.redo_button.clicked.connect(self.undo_stack.redo)
        toolbar.addWidget(self.redo_button)
        self.undo_stack.canUndoChanged.connect(self.undo_button.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.redo_button.setEnabled)

        self.addToolBarBreak()
        layoutbar = QToolBar("격자 / 정렬", self)
        layoutbar.setMovable(False)
        self.addToolBar(layoutbar)
        self.grid_button = QPushButton("격자 표시")
        self.grid_button.setCheckable(True)
        self.grid_button.setChecked(True)
        self.grid_button.toggled.connect(self.toggle_grid)
        layoutbar.addWidget(self.grid_button)
        self.snap_grid_button = QPushButton("격자 맞춤")
        self.snap_grid_button.setCheckable(True)
        self.snap_grid_button.setChecked(False)
        self.snap_grid_button.setToolTip(
            "켜면 노드 생성 및 이동 시 가장 가까운 격자 교차점에 자동 정렬. "
            "격자 표시와는 독립적이며 스마트 가이드보다 우선합니다."
        )
        self.snap_grid_button.toggled.connect(self.toggle_snap_to_grid)
        layoutbar.addWidget(self.snap_grid_button)
        self.guide_button = QPushButton("스마트 가이드")
        self.guide_button.setCheckable(True)
        self.guide_button.setChecked(True)
        self.guide_button.setToolTip("노드 드래그 중 다른 노드와 X 또는 Y가 비슷해지면 자동 정렬")
        self.guide_button.toggled.connect(self.toggle_smart_guides)
        layoutbar.addWidget(self.guide_button)
        for caption, method in (
            ("선택 노드 수평 정렬", lambda: self.align_selected("y")),
            ("선택 노드 수직 정렬", lambda: self.align_selected("x")),
        ):
            button = QPushButton(caption)
            button.clicked.connect(method)
            layoutbar.addWidget(button)
        layoutbar.addSeparator()
        layoutbar.addWidget(QLabel("  눈금 척도: "))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setDecimals(2)
        self.scale_spin.setRange(0, 1000)
        self.scale_spin.setSingleStep(0.5)
        self.scale_spin.setSuffix(" m/칸")
        self.scale_spin.setSpecialValueText("척도 없음")
        self.scale_spin.setToolTip("0은 도식 배치만 의미합니다. 실측값에 맞춰 보정했다면 1칸의 미터 수를 입력하세요.")
        self.scale_spin.valueChanged.connect(self.update_scale)
        layoutbar.addWidget(self.scale_spin)

        QShortcut(QKeySequence.StandardKey.Save, self, activated=self.save_map)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_stack.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.undo_stack.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self.undo_stack.redo)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.delete_selected)
        self.set_mode("select")
        self.update_floor_choices(prefer_index=True)
        self.render_floor()
        self.statusBar().showMessage("2D 여러 층 탭에서 층별 노드를 동시에 보고 연결할 수 있습니다.")

    def snapshot(self) -> dict:
        """Capture only editable map data (not screen widget objects)."""
        return copy.deepcopy({
            "floors": self.floors,
            "nodes": self.nodes,
            "edges": self.edges,
            "entrances": self.entrances,
            "next_node_uid": self.next_node_uid,
            "next_edge_id": self.next_edge_id,
            "next_entrance_id": self.next_entrance_id,
            "meters_per_grid": self.meters_per_grid,
            "active_floor": self.floor_combo.currentText(),
        })

    def record_edit(self, description: str, before: dict):
        """Commit exactly one undo command for one completed user edit."""
        after = self.snapshot()
        if before != after:
            self.undo_stack.push(MapHistoryCommand(self, description, before, after))

    def restore_snapshot(self, state: dict):
        """Restore model and refresh GUI with signals blocked to avoid new edits."""
        state = copy.deepcopy(state)
        self.floors = state["floors"]
        self.nodes = state["nodes"]
        self.edges = state["edges"]
        self.entrances = state.get("entrances", [])
        self.next_node_uid = state["next_node_uid"]
        self.next_edge_id = state["next_edge_id"]
        self.next_entrance_id = state.get("next_entrance_id", 1)
        self.meters_per_grid = state["meters_per_grid"]
        floor = state["active_floor"]
        if floor not in self.floors:
            floor = self.floors[0]
        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        self.floor_combo.addItems(self.floors)
        self.floor_combo.setCurrentText(floor)
        self.floor_combo.blockSignals(False)
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(self.meters_per_grid)
        self.scale_spin.blockSignals(False)
        self.first_endpoint = None
        self.dragging_node_id = None
        self.drag_start_snapshot = None
        self.update_floor_choices()
        self.render_floor()

    def begin_node_drag(self):
        """Take a single snapshot at drag start, not once per mouse move."""
        self.drag_start_snapshot = self.snapshot()

    def finish_node_drag(self):
        before = self.drag_start_snapshot
        self.drag_start_snapshot = None
        if before is not None:
            self.record_edit("노드 이동", before)
            self.render_floor()  # Redraw other panes showing the same floor.

    def find_edge(self, edge_id: str) -> dict | None:
        return next((edge for edge in self.edges if edge["id"] == edge_id), None)

    def find_entrance(self, entrance_id: str) -> dict | None:
        return next((item for item in self.entrances if item["id"] == entrance_id), None)

    def node_name(self, node_id: str) -> str:
        return self.nodes[node_id].get("node_id", "") or f"(이름 없음:{node_id})"

    def set_mode(self, mode: str):
        self.mode = mode
        self.first_endpoint = None
        for key, button in self.mode_buttons.items():
            button.setChecked(key == mode)
        self.statusBar().showMessage({
            "select": "선택: 노드 드래그 / 노드 더블클릭으로 이름 편집 / 엣지 더블클릭으로 실측 길이",
            "add": "노드 추가: 빈 캔버스 클릭 (이름은 나중에 직접 입력)",
            "connect": "엣지 연결: 같은 층 또는 서로 다른 층의 노드 두 개를 차례로 클릭",
            "entrance": "Entrance 추가: 연결할 노드를 클릭 → 출입구 방향의 빈 공간 클릭 (E1, E2 자동 번호)",
        }[mode])

    def create_node(self, pos: QPointF, floor: str | None = None):
        before = self.snapshot()
        while f"U{self.next_node_uid:04d}" in self.nodes:
            self.next_node_uid += 1
        internal_id = f"U{self.next_node_uid:04d}"
        self.next_node_uid += 1
        # Snap newly created nodes only when the independent toggle is ON.
        if self.snap_to_grid_enabled:
            pos = self.snap_point_to_grid(pos)
        x, y = float(pos.x()), float(pos.y())
        self.nodes[internal_id] = {
            "id": internal_id, "node_id": "",
            "floor": floor or self.floor_combo.currentText(),
            "x": round(x, 2), "y": round(y, 2), "type": "intersection",
        }
        self.render_floor()
        self.record_edit("노드 추가", before)
        self.statusBar().showMessage("이름 없는 노드 생성. '선택 / 이동'으로 전환 후 더블클릭해서 이름 입력")

    def edit_node_label(self, node_id: str):
        node = self.nodes[node_id]
        current = node.get("node_id", "")
        label, ok = QInputDialog.getText(self, "노드 이름 입력", "표시할 노드 ID (예: N12):", text=current)
        if not ok:
            return
        label = label.strip()
        if label and any(other_id != node_id and other.get("node_id", "").casefold() == label.casefold()
                         for other_id, other in self.nodes.items()):
            QMessageBox.warning(self, "이름 중복", f"이미 '{label}' 이름을 사용하는 노드가 있습니다.")
            return
        if label == current:
            return
        before = self.snapshot()
        node["node_id"] = label  # Internal ID and edges remain unchanged.
        for canvas in self.all_canvases():
            item = canvas.node_items.get(node_id)
            if item is not None:
                item.refresh_label()
        self.record_edit("노드 이름 변경", before)
        self.update_window_title()
        self.statusBar().showMessage(f"노드 이름 변경: {label or '(미지정)'}")

    def create_entrance(self, node_uid: str, direction_point: QPointF):
        """Attach an external entry marker to a node (NOT to the routing graph)."""
        node = self.nodes.get(node_uid)
        if node is None:
            return
        if not node.get("node_id"):
            QMessageBox.warning(self, "Entrance 추가", "먼저 노드 이름(N1 등)을 정하세요.")
            return
        dx = float(direction_point.x()) - float(node["x"])
        dy = float(direction_point.y()) - float(node["y"])
        norm = math.hypot(dx, dy)
        if norm < 8.0:
            self.statusBar().showMessage("노드 중심에서 조금 떨어진 방향을 클릭하세요.")
            return
        before = self.snapshot()
        used = {entry["id"].casefold() for entry in self.entrances}
        while f"e{self.next_entrance_id}" in used:
            self.next_entrance_id += 1
        eid = f"E{self.next_entrance_id}"
        self.next_entrance_id += 1
        self.entrances.append({
            "id": eid,
            "node": node_uid,
            "dx": round(dx / norm, 8),
            "dy": round(dy / norm, 8),
        })
        self.render_floor()
        self.record_edit(f"Entrance {eid} 추가", before)
        self.statusBar().showMessage(
            f"{eid} 추가 → {self.node_name(node_uid)} / from_{eid}. "
            "DB 폴더는 build_database.py --prepare-folders로 생성됩니다."
        )

    def edit_entrance(self, entrance_id: str):
        entry = self.find_entrance(entrance_id)
        if entry is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"외부 진입점 {entrance_id} 편집")
        form = QFormLayout(dialog)
        name = QLineEdit(entry["id"])
        name.setToolTip("E1, E2와 같이 E + 양의 정수")
        form.addRow("Entrance ID", name)
        angle = QDoubleSpinBox()
        angle.setRange(0, 359.9)
        angle.setDecimals(1)
        angle.setSingleStep(5)
        angle.setSuffix("°")
        angle.setValue(math.degrees(math.atan2(entry["dy"], entry["dx"])) % 360)
        form.addRow("방향 (→0°, ↓90°)", angle)
        form.addRow(QLabel("빨간 선은 실측 Edge가 아닌 외부 진입 방향 표시입니다."))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_id = name.text().strip().upper()
        if not re.fullmatch(r"E[1-9][0-9]*", new_id):
            QMessageBox.warning(self, "Entrance ID", "E1, E2, E3 ... 형식으로 입력하세요.")
            return
        if any(other is not entry and other["id"].casefold() == new_id.casefold()
               for other in self.entrances):
            QMessageBox.warning(self, "Entrance ID", f"이미 {new_id}가 존재합니다.")
            return
        rad = math.radians(angle.value())
        dx, dy = round(math.cos(rad), 8), round(math.sin(rad), 8)
        if (new_id, dx, dy) == (entry["id"], entry["dx"], entry["dy"]):
            return
        before = self.snapshot()
        old_id = entry["id"]
        entry.update({"id": new_id, "dx": dx, "dy": dy})
        self.next_entrance_id = max(self.next_entrance_id, int(new_id[1:]) + 1)
        self.render_floor()
        self.record_edit("Entrance 번호/방향 수정", before)
        if new_id != old_id:
            self.statusBar().showMessage(
                f"{old_id} → {new_id}: 이미 촬영된 from_{old_id} 폴더는 "
                f"직접 from_{new_id}로 옮겨야 합니다."
            )
        else:
            self.statusBar().showMessage(f"{new_id} 진입 방향 수정 완료")

    def select_endpoint(self, node_id: str):
        if self.first_endpoint is None:
            self.first_endpoint = node_id
            self.statusBar().showMessage(f"{self.node_name(node_id)} 선택. 연결할 두 번째 노드를 클릭하세요.")
            return
        first = self.first_endpoint
        self.first_endpoint = None
        if first != node_id:
            if self.nodes[first]["floor"] == self.nodes[node_id]["floor"]:
                self.add_edge(first, node_id, "corridor")
            else:
                kind, ok = QInputDialog.getItem(
                    self, "층간 연결", "연결 유형:", ["stairs", "elevator"], 0, False
                )
                if ok:
                    self.add_edge(first, node_id, kind)
                else:
                    self.statusBar().showMessage("층간 연결 취소")

    def add_edge(self, source: str, target: str, kind: str):
        if source not in self.nodes or target not in self.nodes or source == target:
            return
        for edge in self.edges:
            if {edge["from"], edge["to"]} == {source, target} and edge["type"] == kind:
                self.statusBar().showMessage("이미 존재하는 연결입니다.")
                return
        before = self.snapshot()
        edge_id = f"E{self.next_edge_id}"
        self.next_edge_id += 1
        self.edges.append({
            "id": edge_id, "from": source, "to": target,
            "type": kind, "bidirectional": True, "length_m": None,
        })
        self.render_floor()
        self.record_edit("엣지 연결", before)
        self.statusBar().showMessage(f"{self.node_name(source)} ↔ {self.node_name(target)} 연결 완료")

    def edit_edge_length(self, edge_id: str):
        edge = self.find_edge(edge_id)
        if edge is None:
            return
        initial = edge.get("length_m") or 0
        value, ok = QInputDialog.getDouble(
            self, "Edge 실측 길이", "실측 길이 (m), 0 = 미설정:",
            float(initial), 0, 100000, 2,
        )
        if ok:
            new_length = value if value > 0 else None
            if edge.get("length_m") == new_length:
                return
            before = self.snapshot()
            edge["length_m"] = new_length
            for canvas in self.all_canvases():
                item = canvas.edge_items.get(edge_id)
                if item is not None:
                    item.update_line()
            self.record_edit("엣지 길이 수정", before)

    def toggle_grid(self, enabled: bool):
        self.grid_visible = enabled
        for canvas in self.all_canvases():
            canvas.viewport().update()

    @staticmethod
    def snap_point_to_grid(position: QPointF) -> QPointF:
        """Round canvas coordinates to the nearest GRID intersection."""
        return QPointF(
            round(position.x() / GRID) * GRID,
            round(position.y() / GRID) * GRID,
        )

    def toggle_snap_to_grid(self, enabled: bool):
        self.snap_to_grid_enabled = enabled
        self.clear_alignment_guides()
        self.statusBar().showMessage(
            "격자 맞춤 ON: 노드 생성/이동 시 격자점에 정렬 (스마트 가이드보다 우선)"
            if enabled else "격자 맞춤 OFF: 자유 배치 / 스마트 가이드 사용 가능"
        )

    def clear_alignment_guides(self):
        self.guide_x = None
        self.guide_y = None
        for canvas in self.all_canvases():
            canvas.guide_x = None
            canvas.guide_y = None
            canvas.viewport().update()

    def toggle_smart_guides(self, enabled: bool):
        self.smart_guides_enabled = enabled
        self.clear_alignment_guides()
        self.statusBar().showMessage("스마트 가이드 ON" if enabled else "스마트 가이드 OFF")

    def align_to_other_nodes(self, moving_id: str, position: QPointF,
                             canvas: Canvas) -> QPointF:
        """Align moving node's center to nearby node centers, independently on X/Y.

        All candidates are from the active floor. Tolerance stays constant on
        screen even if the user zooms in or out.
        """
        zoom = max(abs(canvas.transform().m11()), 0.01)
        tolerance = SMART_GUIDE_TOLERANCE_PX / zoom
        x, y = float(position.x()), float(position.y())
        closest_x = closest_y = None
        min_dx = min_dy = tolerance

        for other_id, other_item in canvas.node_items.items():
            if other_id == moving_id:
                continue
            other_pos = other_item.pos()
            dx = abs(x - other_pos.x())
            dy = abs(y - other_pos.y())
            if dx < min_dx:
                min_dx, closest_x = dx, other_pos.x()
            if dy < min_dy:
                min_dy, closest_y = dy, other_pos.y()

        self.guide_x = closest_x
        self.guide_y = closest_y
        canvas.guide_x = closest_x
        canvas.guide_y = closest_y
        canvas.viewport().update()
        return QPointF(closest_x if closest_x is not None else x,
                       closest_y if closest_y is not None else y)

    def align_selected(self, axis: str):
        items = [item for item in self.active_canvas.scene().selectedItems()
                 if isinstance(item, NodeItem)]
        if len(items) < 2:
            self.statusBar().showMessage("정렬할 노드를 두 개 이상 선택하세요 (Ctrl+클릭 또는 드래그 선택)")
            return
        before = self.snapshot()
        if axis == "x":
            target = sum(item.pos().x() for item in items) / len(items)
        else:
            target = sum(item.pos().y() for item in items) / len(items)
        for item in items:
            x, y = item.pos().x(), item.pos().y()
            item.setPos(QPointF(target, y) if axis == "x" else QPointF(x, target))
        self.record_edit("노드 일괄 정렬", before)
        self.render_floor()
        self.statusBar().showMessage("선택한 노드를 " + ("수직" if axis == "x" else "수평") + " 정렬했습니다.")

    def update_scale(self, value: float):
        before = self.snapshot()
        self.meters_per_grid = float(value)
        for canvas in self.all_canvases():
            for item in canvas.edge_items.values():
                item.update_line()
        self.record_edit("눈금 척도 변경", before)

    def add_floor(self):
        floor, ok = QInputDialog.getText(self, "층 추가", "층 이름 (예: B1, 2F):")
        if not ok or not floor.strip():
            return
        floor = floor.strip()
        if floor not in self.floors:
            before = self.snapshot()
            self.floors.append(floor)
            self.floor_combo.addItem(floor)
            self.floor_combo.setCurrentText(floor)
            self.update_floor_choices(prefer_index=True)
            self.render_floor()
            self.record_edit("층 추가", before)
        else:
            self.floor_combo.setCurrentText(floor)

    def connect_floors(self):
        if len(self.nodes) < 2:
            QMessageBox.information(self, "층간 연결", "먼저 연결할 노드들을 생성하세요.")
            return
        options = [f"{self.node_name(uid)} [{uid}] ({node['floor']})" for uid, node in self.nodes.items()]
        source_text, ok = QInputDialog.getItem(self, "층간 연결", "시작 노드", options, 0, False)
        if not ok:
            return
        target_text, ok = QInputDialog.getItem(self, "층간 연결", "도착 노드", options, 0, False)
        if not ok:
            return
        source = source_text.split("[")[-1].split("]")[0]
        target = target_text.split("[")[-1].split("]")[0]
        if source == target or self.nodes[source]["floor"] == self.nodes[target]["floor"]:
            QMessageBox.warning(self, "층간 연결", "서로 다른 층의 두 노드를 선택하세요.")
            return
        kind, ok = QInputDialog.getItem(self, "연결 유형", "이동 수단", ["stairs", "elevator"], 0, False)
        if ok:
            self.add_edge(source, target, kind)
            self.statusBar().showMessage(f"층간 연결 완료: {self.node_name(source)} ↔ {self.node_name(target)}")

    def all_canvases(self):
        return [self.canvas] + [pane.canvas for pane in self.multi_panes]

    def update_floor_choices(self, prefer_index: bool = False):
        for pane in self.multi_panes:
            pane.set_floors(self.floors, prefer_index=prefer_index)

    def update_pane_visibility(self, _=None):
        count = self.pane_count.currentIndex() + 2
        for i, pane in enumerate(self.multi_panes):
            pane.setVisible(i < count)

    def on_tab_changed(self, index: int):
        if index == 0:
            self.active_canvas = self.canvas
        elif index == 1:
            self.active_canvas = self.multi_panes[0].canvas
        elif index == 2:
            self.ensure_3d_viewer()
            self.refresh_3d()

    def update_adjacent_edges(self, node_id: str, canvas: Canvas):
        """Update only the dragged view; the other panes refresh on release."""
        for edge in self.edges:
            if node_id in (edge["from"], edge["to"]):
                item = canvas.edge_items.get(edge["id"])
                if item is not None:
                    item.update_line()
        for entrance in self.entrances:
            if entrance["node"] == node_id:
                item = canvas.entrance_items.get(entrance["id"])
                if item is not None:
                    item.update_line()

    def render_canvas(self, canvas: Canvas):
        canvas.node_items.clear()
        canvas.edge_items.clear()
        canvas.entrance_items.clear()
        canvas.scene().clear()
        floor = canvas.current_floor()
        for node in self.nodes.values():
            if node["floor"] == floor:
                item = NodeItem(self, node, canvas)
                canvas.scene().addItem(item)
                canvas.node_items[node["id"]] = item
        for edge in self.edges:
            src = canvas.node_items.get(edge["from"])
            dst = canvas.node_items.get(edge["to"])
            if src is not None and dst is not None:
                item = EdgeItem(self, edge, src, dst)
                canvas.scene().addItem(item)
                canvas.edge_items[edge["id"]] = item
        for entrance in self.entrances:
            src = canvas.node_items.get(entrance["node"])
            if src is not None:
                item = EntranceItem(self, entrance, src)
                canvas.scene().addItem(item)
                canvas.entrance_items[entrance["id"]] = item
        for pane in self.multi_panes:
            if pane.canvas is canvas:
                pane.label.setText(f"화면 {pane.index + 1} · {len(canvas.node_items)} nodes")
                break

    def render_floor(self, _=None):
        """Refresh the single view and every simultaneously displayed floor."""
        self.dragging_node_id = None
        self.clear_alignment_guides()
        for canvas in self.all_canvases():
            self.render_canvas(canvas)
        self.update_window_title()
        if self.plotter is not None and self.tabs.currentIndex() == 2:
            self.refresh_3d()

    def update_window_title(self):
        floor = self.floor_combo.currentText()
        unnamed = sum(not node.get("node_id") for node in self.nodes.values())
        self.setWindowTitle(f"Node-VPR · Node Map Editor v0.7 | {floor} | "
                            f"{len(self.nodes)} nodes · {len(self.edges)} edges · "
                            f"{len(self.entrances)} entrances | 이름 미지정 {unnamed}")

    def ensure_3d_viewer(self):
        if self.plotter is not None:
            return
        if QtInteractor is None:
            self.viewer_placeholder.setText(
                f"3D 라이브러리 로드 실패: {PV_ERROR}\n"
                "가상환경에 pyvista pyvistaqt PySide6가 설치되어 있는지 확인하세요."
            )
            return
        try:
            self.plotter = QtInteractor(self.view_3d_page)
            self.plotter.set_background("#f7f9fc")
            self.view_3d_layout.removeWidget(self.viewer_placeholder)
            self.viewer_placeholder.hide()
            self.view_3d_layout.addWidget(self.plotter, stretch=1)
        except Exception as exc:
            self.plotter = None
            self.viewer_placeholder.setText(f"3D 뷰어 초기화 실패: {exc}")

    def _floor_elevations(self) -> dict[str, float]:
        """Schematic height, NOT a surveyed floor elevation."""
        results = {}
        nonstandard = 0
        for floor in self.floors:
            label = floor.strip().upper()
            basement = re.fullmatch(r"B(\d+)(?:F)?", label)
            above = re.fullmatch(r"(\d+)F", label)
            if basement:
                results[floor] = -6.0 * int(basement.group(1))
            elif above:
                results[floor] = 6.0 * (int(above.group(1)) - 1)
            else:
                results[floor] = 6.0 * (len(self.floors) + nonstandard)
                nonstandard += 1
        return results

    def refresh_3d(self):
        """Render all floors and cross-floor links using the same node/edge JSON."""
        if self.plotter is None:
            return
        elevations = self._floor_elevations()
        pl = self.plotter
        try:
            pl.clear()
            pl.set_background("#f7f9fc")
            # Canvas units are schematic. Map 25 canvas pixels to 1 render unit.
            points = {
                uid: (float(n["x"]) / GRID,
                      -float(n["y"]) / GRID,
                      elevations[n["floor"]])
                for uid, n in self.nodes.items()
            }
            if points:
                xs = [pt[0] for pt in points.values()]
                ys = [pt[1] for pt in points.values()]
                xlo, xhi = min(xs), max(xs)
                ylo, yhi = min(ys), max(ys)
            else:
                xlo, xhi, ylo, yhi = -10.0, 10.0, -8.0, 8.0
            padding = 2.8
            width = max(xhi - xlo + 2 * padding, 12.0)
            height = max(yhi - ylo + 2 * padding, 10.0)
            cx, cy = (xhi + xlo) / 2, (yhi + ylo) / 2

            for index, floor in enumerate(self.floors):
                z = elevations[floor]
                plate = pv.Plane(center=(cx, cy, z - 0.3),
                                 direction=(0, 0, 1),
                                 i_size=width, j_size=height)
                tint = "#89a8c2" if index % 2 == 0 else "#9eb6ae"
                pl.add_mesh(plate, color=tint, opacity=0.12,
                            show_edges=True, edge_color="#9baec0",
                            name=f"floor_{index}")
                label_point = np.array([[cx - width / 2, cy - height / 2, z]])
                pl.add_point_labels(label_point, [floor], show_points=False,
                                    font_size=14, bold=True, text_color="#2c4d65",
                                    shape=None, always_visible=True,
                                    name=f"label_floor_{index}")

            # Draw connections after floor plates so cross-floor links remain visible.
            for edge in self.edges:
                a, b = edge["from"], edge["to"]
                if a not in points or b not in points:
                    continue
                kind = edge.get("type", "corridor")
                color = {"corridor": "#536b91", "stairs": "#df8b33",
                         "elevator": "#17967e"}.get(kind, "#66768e")
                pl.add_mesh(pv.Line(points[a], points[b]), color=color,
                            line_width=4 if kind != "corridor" else 3,
                            render_lines_as_tubes=True)

            # Entrance markers are visual-only red spurs on their floor.
            entrance_tips = []
            entrance_labels = []
            for entry in self.entrances:
                src = points.get(entry["node"])
                if src is None:
                    continue
                dx, dy = float(entry["dx"]), float(entry["dy"])
                length = math.hypot(dx, dy)
                if length < 1e-9:
                    continue
                tip = (src[0] + (ENTRANCE_LENGTH / GRID) * dx / length,
                       src[1] - (ENTRANCE_LENGTH / GRID) * dy / length,
                       src[2])
                pl.add_mesh(pv.Line(src, tip), color=ENTRANCE_COLOR,
                            line_width=5, render_lines_as_tubes=True)
                entrance_tips.append(tip)
                entrance_labels.append(entry["id"])
            if entrance_tips:
                tips = np.asarray(entrance_tips, dtype=float)
                pl.add_points(tips, color=ENTRANCE_COLOR, point_size=12,
                              render_points_as_spheres=True)
                pl.add_point_labels(tips, entrance_labels, show_points=False,
                                    font_size=11, text_color=ENTRANCE_COLOR,
                                    shape="rounded_rect", shape_color="#ffffff",
                                    shape_opacity=0.85, always_visible=True)

            if points:
                point_array = np.array(list(points.values()), dtype=float)
                pl.add_points(point_array, color="#2563eb",
                              point_size=16, render_points_as_spheres=True)
                labels = [self.node_name(uid) for uid in points]
                pl.add_point_labels(point_array, labels,
                                    show_points=False, font_size=11,
                                    text_color="#12263a", shape="rounded_rect",
                                    shape_color="#ffffff", shape_opacity=0.78,
                                    always_visible=True)
            pl.add_axes()
            pl.reset_camera()
            pl.render()
        except Exception as exc:
            self.statusBar().showMessage(f"3D 지도 렌더링 오류: {exc}")

    def closeEvent(self, event):
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass
        super().closeEvent(event)

    def delete_selected(self):
        selected = self.active_canvas.scene().selectedItems()
        node_ids = {item.node_id for item in selected if isinstance(item, NodeItem)}
        edge_ids = {item.edge_id for item in selected if isinstance(item, EdgeItem)}
        entrance_ids = {item.entrance_id for item in selected
                        if isinstance(item, EntranceItem)}
        if not node_ids and not edge_ids and not entrance_ids:
            return
        before = self.snapshot()
        for node_id in node_ids:
            self.nodes.pop(node_id, None)
        self.edges = [edge for edge in self.edges if edge["id"] not in edge_ids
                      and edge["from"] not in node_ids and edge["to"] not in node_ids]
        self.entrances = [entry for entry in self.entrances
                          if entry["id"] not in entrance_ids and entry["node"] not in node_ids]
        self.render_floor()
        self.record_edit("선택 항목 삭제", before)

    def save_map(self):
        if self.current_path is None:
            filename, _ = QFileDialog.getSaveFileName(self, "지도 저장", "node_map.json", "JSON (*.json)")
            if not filename:
                return
            self.current_path = Path(filename)
        payload = {
            "version": 3,
            "coordinate_system": "schematic_layout",
            "grid_step_canvas": GRID,
            "meters_per_grid": self.meters_per_grid,
            "floors": self.floors,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "entrances": self.entrances,
            "next_node_uid": self.next_node_uid,
            "next_edge_id": self.next_edge_id,
            "next_entrance_id": self.next_entrance_id,
        }
        try:
            self.current_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "저장 실패", str(exc))
            return
        self.statusBar().showMessage(f"저장 완료: {self.current_path}")

    def load_map(self):
        filename, _ = QFileDialog.getOpenFileName(self, "지도 불러오기", "", "JSON (*.json)")
        if not filename:
            return
        try:
            data = json.loads(Path(filename).read_text(encoding="utf-8"))
            floors = list(data["floors"])
            node_list = list(data["nodes"])
            edges = list(data["edges"])
            entrances = list(data.get("entrances", []))
            nodes = {}
            labels = set()
            for source_node in node_list:
                node = dict(source_node)
                uid = str(node["id"])
                if uid in nodes:
                    raise ValueError(f"중복된 내부 ID: {uid}")
                # v0.1 used the internal ID as the visible label.
                node.setdefault("node_id", node.get("label", uid if int(data.get("version", 1)) < 2 else ""))
                label = str(node["node_id"]).strip()
                if label and label.casefold() in labels:
                    raise ValueError(f"중복된 노드 이름: {label}")
                if label:
                    labels.add(label.casefold())
                if node["floor"] not in floors:
                    raise ValueError(f"존재하지 않는 층: {node['floor']}")
                nodes[uid] = node
            if not floors:
                raise ValueError("층이 비어 있습니다.")
            if any(edge["from"] not in nodes or edge["to"] not in nodes for edge in edges):
                raise ValueError("존재하지 않는 노드를 참조하는 엣지가 있습니다.")
            for edge in edges:
                edge.setdefault("length_m", None)
                if edge["length_m"] is not None and float(edge["length_m"]) < 0:
                    raise ValueError("Edge 길이는 음수가 될 수 없습니다.")
            next_uid = int(data.get("next_node_uid", 1))
            next_eid = max(int(data.get("next_edge_id", 1)),
                           max((int(e["id"][1:]) + 1 for e in edges
                                if e["id"].startswith("E") and e["id"][1:].isdigit()), default=1))
            seen_ent = set()
            max_ent_num = 0
            for entry in entrances:
                eid = str(entry["id"]).strip()
                if not re.fullmatch(r"E[1-9][0-9]*", eid):
                    raise ValueError(f"외부 진입점 ID 오류: {eid}")
                if eid.casefold() in seen_ent:
                    raise ValueError(f"중복된 외부 진입점 ID: {eid}")
                seen_ent.add(eid.casefold())
                if entry["node"] not in nodes:
                    raise ValueError(f"{eid}: 존재하지 않는 노드에 연결됨")
                dx, dy = float(entry["dx"]), float(entry["dy"])
                if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) < 1e-9:
                    raise ValueError(f"{eid}: 잘못된 방향 벡터")
                max_ent_num = max(max_ent_num, int(eid[1:]))
            next_entrance_id = max(int(data.get("next_entrance_id", 1)), max_ent_num + 1)
            scale = max(0.0, float(data.get("meters_per_grid", 0.0)))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "불러오기 실패", f"지도 데이터를 확인하세요.\n{exc}")
            return
        self.floors, self.nodes, self.edges = floors, nodes, edges
        self.entrances = entrances
        self.next_node_uid, self.next_edge_id = next_uid, next_eid
        self.next_entrance_id = next_entrance_id
        self.current_path = Path(filename)
        self.meters_per_grid = scale
        self.floor_combo.blockSignals(True)
        self.floor_combo.clear()
        self.floor_combo.addItems(floors)
        self.floor_combo.blockSignals(False)
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)
        self.update_floor_choices(prefer_index=True)
        self.render_floor()
        self.set_mode("select")
        # Opening another document starts an independent undo history.
        self.undo_stack.clear()
        self.statusBar().showMessage(f"불러오기 완료: {filename}")


def main():
    app = QApplication([])
    window = MapEditor()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
