from deep_sort.application_util.visualization import cv2
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from mobile_net import ObjectRecognition

file_path = "videos/chhh.MOV"
cap = cv2.VideoCapture(file_path)

model = ObjectRecognition()
encoder = init_encoder()
config = DeepSORTConfig()

    

while(True):
    ret, frame = cap.read()
    boxes = model.get_boxes(frame)
    print(boxes)
    if len(boxes) > 0:
        encoding = generate_detections(encoder, boxes, frame)
        tracked_ids,detections = run_deep_sort(frame, encoding, config)
        for i in range(len(tracked_ids)):
            if not tracked_ids[i].is_confirmed() or tracked_ids[i].time_since_update > 0:
                continue
            print(tracked_ids[i].track_id, tracked_ids[i].to_tlwh())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()