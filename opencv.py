import cv2


def main():
    # Load the pre-trained model and image
    model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
    image = cv2.imread('coffee_cup.jpg')

    # Preprocess the image and pass it through the model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    # Loop over the detections and draw bounding boxes
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def main2():
    # Load the image
    image = cv2.imread('datasets/Hamilton/with_wells/1a1b9344d8c95335668691e0d4f83eae.png')

    # Apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Show the original and blurred images
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Blurred Image', blurred_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def edge_detect():
    # Load the image
    image = cv2.imread('datasets/Hamilton/with_wells/1a1b9344d8c95335668691e0d4f83eae.png')


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def feature_detect():
    # Load the image
    image = cv2.imread('datasets/Hamilton/with_wells/1a1b9344d8c95335668691e0d4f83eae.png')

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

    cv2.imshow('ORB Keypoints', image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def circle_detect():
    import numpy as np
    import webcolors
    from scipy.spatial import KDTree

    mapped_cell = {
        "well_name": "",
        "well_coordinates": None,
        "well_color": None
    }
    mapped_cells_list = []
    sorted_outer_list = []
    sorted_final_list = []
    row_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Get Color names from RGB values
    def convert_rgb_to_color_name(rgb_input):
        hexnames = webcolors.css3_hex_to_names
        names = []
        positions = []

        for hex, name in hexnames.items():
            names.append(name)
            positions.append(webcolors.hex_to_rgb(hex))

        spacedb = KDTree(positions)

        querycolor = rgb_input
        dist, index = spacedb.query(querycolor)
        return names[index]



    # Load the image
    img = cv2.imread('datasets/Hamilton/with_wells/1a1b9344d8c95335668691e0d4f83eae.png')

    img_height, img_width, img_col_channels = img.shape # General image data
    print(img_height, img_width)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blurred_image = cv2.GaussianBlur(gray_img, (9, 9), 2)

    # gray_img = cv2.blur(gray_img, (3,3)) # blur to reduce noise?

    detected_circles = cv2.HoughCircles(gray_img, # image for detection
                                        cv2.HOUGH_GRADIENT, # detection method
                                        1, # inverse ration of resolution
                                        img_height/64, # Not sure need adjustment
                                        param1 = 200,
                                        param2 = 10,
                                        minRadius = 14, # If unknown input 0
                                        maxRadius = 15 # If unknown input 0
                                    )

    if detected_circles is not None:
        print("Detected") 
        detected_circles = np.uint16(np.around(detected_circles))
        for i in detected_circles[0, :96]: # define maximun number of circles is 96
            cir_center = (i[0], i[1]) # circle center tuple
            # print(cir_center)
            cir_radius = i[2] # circle radius
            cv2.circle(img, cir_center, 1, (0, 0, 255), 5) # circle center col red
            cv2.circle(img, cir_center, cir_radius, (0, 0, 0), 5) # circle outline
        cv2.imshow("Detected Circle", img) 
        cv2.waitKey(0) 
    else:
        print("No circles are detected.")

    cv2.destroyAllWindows()


def feature_detect_with_model():
    # Load the image
    image = cv2.imread('datasets/Hamilton/with_wells/1a1b9344d8c95335668691e0d4f83eae.png')

    net = cv2.dnn.readNetFromONNX('model.onnx')

    # Preprocess and pass the image through the network
    blob = cv2.dnn.blobFromImage(image, 1.0, (256, 256))
    net.setInput(blob)
    detections = net.forward()


def livestream_detect():
    import cv2
    import cvlib as cv
    from cvlib.object_detection import draw_bbox


    video = cv2.VideoCapture(1)
    labels = []

    while True:
        ret, frame = video.read()
        # Bounding box.
        # the cvlib library has learned some basic objects using object learning
        # usually it takes around 800 images for it to learn what a phone is.
        # bbox, label, conf = cv.detect_common_objects(frame)

        # output_image = draw_bbox(frame, bbox, label, conf)

        cv2.imshow("Detection", frame)

        # for item in label:
        #     if item in labels:
        #         pass
        #     else:
        #         labels.append(item)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    # main()
    # main2()
    # edge_detect()
    # feature_detect()
    circle_detect()
    # livestream_detect()