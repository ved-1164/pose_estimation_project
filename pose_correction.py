from utils import calculate_angle

def check_arm_straight(landmarks):

    shoulder = landmarks[11]
    elbow = landmarks[13]
    wrist = landmarks[15]

    angle = calculate_angle(shoulder, elbow, wrist)

    if angle > 170:
        return True
    else:
        return False
    