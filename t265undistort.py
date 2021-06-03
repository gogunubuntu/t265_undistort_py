import pyrealsense2 as rs
# Import OpenCV and numpy
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from math import tan, pi

class t265undistort : 
    def __init__(self) : 
        self.pipe = rs.pipeline()
        cfg = rs.config()
        self.pipe.start(cfg)


        window_size = 5
        min_disp = 0

        num_disp = 112 - min_disp
        self.max_disp = min_disp + num_disp

        profiles = self.pipe.get_active_profile()
        streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
        intrinsics = {"left"  : streams["left"].get_intrinsics(),
                    "right" : streams["right"].get_intrinsics()}

        print("Left camera:",  intrinsics["left"])
        print("Right camera:", intrinsics["right"])

        K_left  = self.camera_matrix(intrinsics["left"])
        D_left  = self.fisheye_distortion(intrinsics["left"])
        K_right = self.camera_matrix(intrinsics["right"])
        D_right = self.fisheye_distortion(intrinsics["right"])
        (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

        (R, T) = self.get_extrinsics(streams["left"], streams["right"])

        stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
        stereo_height_px = 1024         # 300x300 pixel stereo output
        stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

        R_left = np.eye(3)
        R_right = R

        stereo_width_px = stereo_height_px + self.max_disp
        stereo_size = (stereo_width_px, stereo_height_px)
        stereo_cx = (stereo_height_px - 1)/2 + self.max_disp
        stereo_cy = (stereo_height_px - 1)/2

        P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                        [0, stereo_focal_px, stereo_cy, 0],
                        [0,               0,         1, 0]])
        P_right = P_left.copy()
        P_right[0][3] = T[0]*stereo_focal_px

        m1type = cv2.CV_32FC1
        (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
        (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
        self.undistort_rectify = {"left"  : (lm1, lm2),
                            "right" : (rm1, rm2)}
        self.frame_data = {"left"  : None,
                "right" : None,
                "timestamp_ms" : None
                }
    def get_extrinsics(self, src, dst):
        extrinsics = src.get_extrinsics_to(dst)
        R = np.reshape(extrinsics.rotation, [3,3]).T
        T = np.array(extrinsics.translation)
        return (R, T)


        
    def camera_matrix(self, intrinsics):
        return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                        [            0, intrinsics.fy, intrinsics.ppy],
                        [            0,             0,              1]])

    def fisheye_distortion(self, intrinsics):
        return np.array(intrinsics.coeffs[:4])
    
    def get_frame(self) : 
        frame = self.pipe.wait_for_frames()

        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            left_data = np.asanyarray(f1.get_data())
            right_data = np.asanyarray(f2.get_data())
            ts = frameset.get_timestamp()
            self.frame_data["left"] = left_data
            self.frame_data["right"] = right_data
            self.frame_data["timestamp_ms"] = ts


        frame_copy = {"left"  : self.frame_data["left"].copy(),
                      "right" : self.frame_data["right"].copy()}

        center_undistorted={"left" : cv2.remap(src = frame_copy["left"],
                                          map1 = self.undistort_rectify["left"][0],
                                          map2 = self.undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                            "right" : cv2.remap(src = frame_copy["right"],
                                          map1 = self.undistort_rectify["right"][0],
                                          map2 = self.undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}
        left_image  = cv2.cvtColor(center_undistorted["left"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)
        right_image = cv2.cvtColor(center_undistorted["right"][:,self.max_disp:], cv2.COLOR_GRAY2RGB)
        #print(left_image.shape)
        
        return left_image, right_image
        # Set up a mutex to share data between threads
    def stop(self) : 
        self.__del__()
    def __del__(self) : 
        self.pipe.stop()
if __name__ == '__main__' : 
    WINDOW_TITLE = 'Realsense'
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    t265 = t265undistort()
    while True : 
        lim, rim = t265.get_frame()
        cv2.imshow(WINDOW_TITLE, np.hstack((lim, rim)))
        key = cv2.waitKey(1)
        if key == ord('q') : 
            cv2.destroyAllWindows()
            pipe.stop()
