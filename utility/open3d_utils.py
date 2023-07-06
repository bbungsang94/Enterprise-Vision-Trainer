import open3d as o3d


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
