import math

def calculate_angle(a, b, c):
    """Calculate angle at point B given three points (a, b, c)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba * mag_bc == 0:
        return 0  # Avoid division by zero

    angle = math.acos(dot_product / (mag_ba * mag_bc))
    return math.degrees(angle)

def count_extended_fingers(hand_landmarks, handedness):
    """Counts extended fingers based on fingertip positions."""
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_pips = [3, 6, 10, 14, 18]  # Lower joints

    count = 0

    # Thumb logic
    if handedness == "Right":
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
            count += 1
    else:
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
            count += 1

    # Other fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y:
            count += 1

    return count