import open3d as o3d


def draw_lines(points, lines=None):
    if lines is None:
        length = len(points)
        lines = [[i, i + 1] for i in range(length - 1)]
        lines.append([length - 1, 0])

    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def get_clicked_point(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 3D 모델을 시각화하여 화면에 보여줍니다.
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    # 사용자의 클릭을 기다립니다.
    print("Please click a point on the 3D model.")
    vis.run()
    view_control = vis.get_view_control()
    point = view_control.convert_click_to_pinhole_camera_coordinates(vis.get_render_option().point_size,
                                                                     vis.get_render_option().point_size,
                                                                     vis.get_window_size()[0] // 2,
                                                                     vis.get_window_size()[1] // 2)
    print(point)
    vis.destroy_window()

    return point
