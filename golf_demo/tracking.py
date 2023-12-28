import numpy as np
import cv2
import os
import time


def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[int(i) - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []

    # We are only inteested in the "sports ball" class
    sportsBallClassId = classes.index("sports ball")

    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if (confidence > confThreshold) and (classId == sportsBallClassId):
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    if (len(indices) > 0):
        # Return the first result
        box = boxes[0]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        return (left, top, width, height)
    else:
        return None


# YOLO parameters
objectnessThreshold = 0.5  # Objectness threshold
confThreshold = 0.5       # Confidence threshold
nmsThreshold = 0.4        # Non-maximum suppression threshold
inpWidth = 416            # Width of network's input image
inpHeight = 416           # Height of network's input image

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Process inputs
videoPath = "golf_1.mp4"

# Open the video file
cap = cv2.VideoCapture(videoPath)

# Read the first frame
hasFrame, frame = cap.read()

# Mode string
mode = "Detecting"
state = "Tracking not started"

# Detected bounding box
ballBoundingBox = None
prev_center = None
prev_time = None

while hasFrame:

    if (mode == "Detecting"):
        # Create a 4D blob from a frame.
        blob = cv2.dnn.blobFromImage(
            frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        # Sets the input to the network
        net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        ballBoundingBox = postprocess(frame, outs)
        if (ballBoundingBox != None):
            # Draw a blue bounding box to indicate detection mode
            p1 = (ballBoundingBox[0], ballBoundingBox[1])
            p2 = (ballBoundingBox[0] + ballBoundingBox[2],
                  ballBoundingBox[1] + ballBoundingBox[3])
            cv2.rectangle(frame, p1, p2, (255, 178, 50), 3)

            # Initialize tracker with frame and bounding box
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, ballBoundingBox)

            # Switch to tracking mode
            mode = "Tracking"
            state = "Tracking started"

            # Display the frame
            cv2.imshow("frame", frame)

            cv2.waitKey(0)

    else:
        # Tracking mode

        # Update tracker
        success, ballBoundingBox = tracker.update(frame)

        if (success):
            # Tracking success
            state = "Tracking success"
            # Calculate the center of the bounding box
            center = (int(ballBoundingBox[0] + 0.5 * ballBoundingBox[2]),
                      int(ballBoundingBox[1] + 0.5 * ballBoundingBox[3]))

            # Draw bounding box
            p1 = (int(ballBoundingBox[0]), int(ballBoundingBox[1]))
            p2 = (int(ballBoundingBox[0] + ballBoundingBox[2]),
                  int(ballBoundingBox[1] + ballBoundingBox[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
            # Calculate speed if there is a previous frame to compare with
            if prev_center is not None and prev_time is not None:
                distance = ((center[0] - prev_center[0])**2 +
                            (center[1] - prev_center[1])**2)**0.5
                time_elapsed = time.time() - prev_time
                # Calculate distance in pixels and convert to meters
                distance_meters = ((center[0] - prev_center[0])**2 +
                                   (center[1] - prev_center[1])**2)**0.5 * 0.1 if prev_center else 0

                # Speed in meters per second
                speed_meters_per_sec = distance_meters / \
                    time_elapsed if time_elapsed > 0 else 0

                # Convert speed to kilometers per hour
                speed_kmph = speed_meters_per_sec * 3.6

                # Display speed information
                cv2.putText(frame, f"Speed: {speed_meters_per_sec:.2f} m/s ({speed_kmph:.2f} km/h)", (0, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 25, 255))

                # Update variables for the next iteration
                prev_center, prev_time = (
                    center, time.time()) if prev_center else (None, None)

                # Speed in pixels per second
                speed = distance / time_elapsed

                # Display speed information
                cv2.putText(frame, f"Speed: {speed:.2f} pixels/second", (0, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            # Update variables for the next iteration
            prev_center = center
            prev_time = time.time()
        else:
            # Tracking failure
            state = "Tracking failure"
            mode = "Detecting"
            tracker = None

    # Draw a black box behind text
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "Mode: "+mode, (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

    color = (255, 255, 255)
    if ("failure" in state):
        color = (0, 0, 255)

    cv2.putText(frame, "State: "+state, (0, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

    # Display the frame
    cv2.imshow("frame", frame)

    # Check for escape key
    key = cv2.waitKey(1)
    if key == 27:
        break

    # Read the next frame
    hasFrame, frame = cap.read()


# Close the video file
cap.release()

# Close all windows
cv2.destroyAllWindows()
