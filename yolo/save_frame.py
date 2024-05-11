import cv2
import random
import argparse
from ultralytics import YOLO

# Add Parser
parser = argparse.ArgumentParser()   
parser.add_argument("--model", type=str, default="yolov8l.pt", help="Model")
parser.add_argument("--source", type=str, default="inference/basketball.mp4", help="Source")
parser.add_argument("--label_size", type=float, default=1, help="Source")
parser.add_argument("--hide_labels", type=bool, default=False, help="Hide Labels")
parser.add_argument("--save_dir", type=str, default="result", help="Save Directory")

args = parser.parse_args()

def draw_box(img, result, class_list, colors) :
    # Get information from result
    xyxy= result.boxes.xyxy.numpy()
    confidence= result.boxes.conf.numpy()
    class_id= result.boxes.cls.numpy().astype(int)
    # Get Class name
    class_name = [class_list[x] for x in class_id]
    # Pack together for easy use
    sum_output = list(zip(class_id, confidence,xyxy))
    # Copy image, in case that we need original image for something
    out_image = img.copy()    

    for run_output in sum_output :
        # Unpack
        label, con, box = run_output
        # Choose color
        box_color = colors[int(label)]
        text_color = (255,255,255)
        # Get Class Name
        label = class_list[int(label)]
        # Draw object box
        first_half_box = (int(box[0]),int(box[1]))
        second_half_box = (int(box[2]),int(box[3]))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
        # Create text
        text_print = '{label} {con:.2f}'.format(label = label, con = con)
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10 ))
        # Get size and baseline
        labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, args.label_size, 1) 

        hideLabels = args.hide_labels
        if(not hideLabels):
            # Draw text's background
            cv2.rectangle(out_image 
                            , (int(box[0]), int(box[1] - labelSize[1] - 10 ))
                            , (int(box[0])+labelSize[0], int(box[1] + baseLine-10))
                            , box_color , cv2.FILLED)        
            # Put text
            cv2.putText(out_image, text_print ,text_location
                        , cv2.FONT_HERSHEY_SIMPLEX , args.label_size
                        , text_color, 1, cv2.LINE_AA)

    return out_image

# Initialize video
cap = cv2.VideoCapture(args.source)

# Initialize YOLOv8 model
model_path = args.model
yolov8_detector = YOLO(model_path)

# Class Name and Colors
label_map = yolov8_detector.names
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]

count = 0
while cap.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue

    # Update object localizer    
    results = yolov8_detector.predict(frame)
    result = results[0].cpu()

    # Draw Detection Results
    combined_img = draw_box(frame, result, label_map, COLORS)        
    
    cv2.imshow("Detected Objects", combined_img)

    # Save dir
    save_dir = args.save_dir
    save_path = save_dir + "/frame-" + str(count) + ".jpg"    

    cv2.imwrite(save_path, combined_img)

    count += 1 
