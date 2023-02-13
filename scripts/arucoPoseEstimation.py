#pylint: disable-all
import pyzed.sl as sl
import numpy as np
import cv2
import rospy
from math import pi
import tf
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

SHOW_IMG = True

# Camea offsets
Z_OFFSET_CAMERA=0.96
Y_OFFSET_CAMERA=0.61
X_OFFSET_CAMERA=0.19

# Dictionary containing ArUco codes
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# 90 degrees rotation needed to match the frames
DEFAULT_ROT = np.array([[0,0,-1],[0,1,0],[1,0,0]])

# specific rotations for each face of the cube
zero = np.matmul(np.array([[1,0,0],[0,1,0],[0,0,1]]),DEFAULT_ROT)
uno = np.matmul(np.array([[1,0,0],[0,-1,0],[0,0,-1]]),DEFAULT_ROT)
due = np.matmul(np.array([[1,0,0],[0,0,1],[0,-1,0]]),DEFAULT_ROT)
tre = np.matmul(np.array([[1,0,0],[0,0,-1],[0,1,0]]),DEFAULT_ROT)
quattro = np.matmul(np.array([[0,0,-1],[0,1,0],[1,0,0]]),DEFAULT_ROT)
cinque = np.matmul(np.array([[0,0,1],[0,1,0],[-1,0,0]]),DEFAULT_ROT)

# Dictionary used to map the ArUco ids to the corresponding rotation
EUL_TRANS_DICT= {
	0 : zero,
	1 : uno,
	2 : due,
    3 : tre,
	4 : quattro,
	5 : cinque
}

# Dictionary used to find the centroid of the cube given a face
CENTER_POINT_OFFSET_DICT={
    0 : np.float32([[-0.025,0,0]]),
    1 : np.float32([[0.025,0,0]]),
    2 : np.float32([[0,0.025,0]]),
    3 : np.float32([[0,-0.025,0]]),
    4 : np.float32([[0,0,0.025]]),
    5 : np.float32([[0,0,-0.025]])
}

# Camera calibration values
cx=958.084
cy=528.676
fx=1077.79
fy=1078.35
k1=-0.049069434790845086
k2=0.0236668632308862
k3=-0.011162880427934998
p1=-0.0005055421561811384
p2=4.10855452151858e-05

# Function used to estimate the pose of the cube from the ArUco position
def pose_estimation(frame, matrix_coefficients, distortion_coefficients):
    
    # Convert the image to a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialization steps
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    # Dection of ArUco code
    corners, ids, rejectedCandidates = detector.detectMarkers(gray)

    # Check if something has been found
    if len(corners) > 0:
        
        # Avoid crashing if the ArUco id is wrong
        if(ids[0] > 5):
            return None,None,None,None
        
        # Choose the lowest id ArUco
        minId = min(ids)[0]
        minIndex = np.argmin(ids)
        
        # Estimate the cube pose given the ArUco code
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[minIndex], 0.04, matrix_coefficients, distortion_coefficients)

        # Draw the ArUco id on the output image
        frame = cv2.putText(frame, 'id: '+str(minId), (int(corners[minIndex][0][0][0]),int(corners[minIndex][0][0][1])), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2, cv2.LINE_AA)
        
        # Transformations needed to have coherent frames
        rmat = cv2.Rodrigues(rvec)[0]
        computed_rtm = np.matmul(rmat,EUL_TRANS_DICT[minId])
        computed_rvec = cv2.Rodrigues(computed_rtm)[0]

        # Draw the frames to the output image
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, computed_rvec, tvec, 0.01)

        # Preliminary operations to find the centroid of the cube
        centroid_offset = CENTER_POINT_OFFSET_DICT[minId]
        homogenous_trans_mtx  = np.append(computed_rtm, [ [tvec[0][0][0]], [tvec[0][0][1]], [tvec[0][0][2]] ], axis=1)
        homogenous_trans_mtx = np.append(homogenous_trans_mtx,[[0,0,0,1]],axis=0)

        # Find x,y position to draw the centroid
        imgpts, jac = cv2.projectPoints(centroid_offset, computed_rvec, tvec, matrix_coefficients, distortion_coefficients)
        imgpts = np.int32(imgpts).reshape(-1,2)
        
        # Find the 3d coordinates of the centroid
        x = CENTER_POINT_OFFSET_DICT[minId][0][0]
        y = CENTER_POINT_OFFSET_DICT[minId][0][1]
        z = CENTER_POINT_OFFSET_DICT[minId][0][2]
        centroid_coords = [ [x], [y], [z], [1] ]
        centroid_coords = np.matmul(homogenous_trans_mtx,centroid_coords)

        # Draw the centroid on the output image
        frame = cv2.circle(frame, (imgpts[0][0], imgpts[0][1]), radius=3, color=(255,0,255), thickness=4)

        return frame, imgpts[0], tvec, computed_rvec

    return None,None,None,None

def main():
    # Start the ROS publisher
    #pub = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)
    pub = rospy.Publisher('/cube_pose', PoseStamped, queue_size=10)

    rospy.init_node('cube_pose_publisher')
    # Initialize the ZED camera
    zed = sl.Camera()
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30
    init_params.depth_minimum_distance = 0.3
    init_params.depth_maximum_distance = 3

    # Open the ZED camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("error")
        exit(-1)
    
    # Image matrix
    image = sl.Mat()

    # Set runtime parameters
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # Camera calibration parameters
    intrinsic_camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    distortion = np.array([k1, k2, p1, p2, k3])

    while not rospy.is_shutdown():

        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT) 
            
            # Convert image from ZED format to OpenCV format
            img = image.get_data()

            # Detection function
            output,center,tvec,rvec = pose_estimation(img, intrinsic_camera, distortion)

            if(output is None):
                continue
            if(center[0]<0 or center[0]>1720 or center[1]<0 or center[1] > 1080):
                print("Out of the image")
                continue

            # Initialize ROS message
            cube_pos = PoseStamped()
            
            # Output image to the screen
            if (SHOW_IMG):
                cv2.imshow('Estimated Pose', output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                   break

            # Populate ROS message
            cube_pos.header =  Header()
            cube_pos.header.frame_id = "map"

            # Cube position
            cube_pos.pose.position.x = -Y_OFFSET_CAMERA + tvec[0][0][2] # x axis is -x axis of camera
            cube_pos.pose.position.y = -X_OFFSET_CAMERA+tvec[0][0][0] # y axis is z axis of camera
            cube_pos.pose.position.z = Z_OFFSET_CAMERA - tvec[0][0][1] # z axis is -y axis of camera

            # Cube orientation
            quaternion = tf.transformations.quaternion_from_euler(rvec[2],rvec[1],-(rvec[0]))
            cube_pos.pose.orientation.x = quaternion[0]
            cube_pos.pose.orientation.y = quaternion[1]
            cube_pos.pose.orientation.z = quaternion[2]
            cube_pos.pose.orientation.w = quaternion[3]

            # Publish ROS message
            pub.publish(cube_pos)
            

if __name__ == "__main__":
    main()
